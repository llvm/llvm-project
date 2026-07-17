#!/usr/bin/env python3
"""Summarize std::init profile violations from a Boost build log.

Reads the plain-text clang log produced by the profiles Boost build
(.github/workflows/profiles-boost-build.yml), classifies each std::init profile
violation by rule kind and by Boost library, and emits:

  * a Markdown report on stdout (for $GITHUB_STEP_SUMMARY): a mermaid pie of the
    rule distribution, a ranked top-libraries table, and a rule x library matrix;
  * an optional summary.json with the full aggregates (--json);
  * an optional self-contained report.html dashboard (--html).

Text parsing is used deliberately: the human-readable log is a required
deliverable and per-TU SARIF capture under parallel b2 is not practical. Every
profile diagnostic ends in "under profile 'std::init'", which isolates our
errors and excludes the test:: profiles (which share some wording) and unrelated
Boost/clang errors. See the diagnostic catalog in DiagnosticSemaKinds.td.

Uses only the Python standard library.
"""

import argparse
import html
import json
import os
import re
import sys
from collections import Counter, defaultdict

# A clang diagnostic line: "path:line:col: error: message".
_DIAG_RE = re.compile(
    r"^(?P<path>[^:\n]+):(?P<line>\d+):(?P<col>\d+):\s+"
    r"(?P<level>error|warning|note):\s+(?P<msg>.*)$"
)
_PROFILE_RE = re.compile(r"under profile '(?P<profile>[^']+)'")
_CRASH_MARKER = "PLEASE submit a bug report"

# Rule classification, applied in order; first match wins. Each entry is
# (rule, predicate(msg)). Ordering matters where messages share substrings
# (e.g. uninit_read's "...read through a '[[ref_to_uninit]]'..." must be caught
# before the ref_to_uninit check, and uninit_write's "does not initialize it"
# before ctor_uninit_member's "constructor does not initialize"). See
# clang/include/clang/Basic/DiagnosticSemaKinds.td for the source messages.
def _has(*needles):
    return lambda m: all(n in m for n in needles)


def _any(*needles):
    return lambda m: any(n in m for n in needles)


_RULES = [
    ("uninit_read", _any("is read before initialization",
                         "accesses uninitialized memory")),
    ("uninit_write", _has("does not initialize it")),
    ("ctor_uninit_member", _has("constructor does not initialize")),
    ("ref_to_uninit", _any("[[ref_to_uninit]]",
                           "binds a reference to uninitialized memory",
                           "binds its implicit object parameter")),
    ("pointer_marker", _has("'[[uninit]]' cannot be applied to a pointer")),
    ("static_marker", _has("'[[uninit]]' cannot be applied to variable",
                           "storage duration")),
    ("union_marker", _has("'[[uninit]]' cannot be applied to", "union")),
    ("uninit_with_initializer",
     _any("cannot be both '[[uninit]]' and have an initializer",
          "default-initialization of its type")),
    ("uninit_decl", _any("must be initialized or marked '[[uninit]]'",
                         "of union type must be initialized")),
    ("static_runtime_init", _has("requires constant initialization")),
]

# Stable order for display (matches _RULES, plus the catch-all).
RULE_ORDER = [name for name, _ in _RULES] + ["other"]

_LIBS_RE = re.compile(r"(?:^|/)libs/([^/]+)/")


def classify_rule(msg):
    for name, pred in _RULES:
        if pred(msg):
            return name
    return "other"


def _lib_from_path_text(path):
    # A compiled-library source: libs/<lib>/... wins (most precise).
    m = _LIBS_RE.search(path)
    if m:
        return m.group(1)
    # A header: the component follows the include root's "boost" directory. Take
    # the segment after the LAST "boost" segment so a doubled root like
    # ".../boost/boost/container/..." (checkout dir + include dir) yields
    # "container", not "boost". Strip the extension for single-file headers
    # (boost/optional.hpp -> optional).
    parts = path.split("/")
    boost_idx = [i for i, p in enumerate(parts) if p == "boost"]
    if boost_idx and boost_idx[-1] + 1 < len(parts):
        comp = parts[boost_idx[-1] + 1]
        return comp.split(".")[0] if "." in comp else comp
    return None


def library_for_path(path, boost_root, cache):
    """Attribute a diagnostic path to a Boost library.

    Headers under boost/ are symlinks into libs/<lib>/include/..., so resolving
    the real path (when boost_root is given) yields precise attribution; fall
    back to the textual boost/<component> heuristic otherwise.
    """
    if path in cache:
        return cache[path]
    candidates = []
    if boost_root:
        for base in (boost_root, os.getcwd()):
            p = path if os.path.isabs(path) else os.path.join(base, path)
            try:
                candidates.append(os.path.realpath(p))
            except OSError:
                pass
    candidates.append(path)
    lib = None
    for cand in candidates:
        lib = _lib_from_path_text(cand)
        if lib:
            break
    lib = lib or "<unknown>"
    cache[path] = lib
    return lib


def parse_log(lines, boost_root=None):
    by_rule = Counter()
    by_library = Counter()
    matrix = defaultdict(Counter)   # library -> rule -> count
    by_file = Counter()
    files = set()
    crash_markers = 0
    non_profile_errors = 0
    cache = {}

    for line in lines:
        if _CRASH_MARKER in line:
            crash_markers += 1
        m = _DIAG_RE.match(line.rstrip("\n"))
        if not m or m.group("level") != "error":
            continue
        msg = m.group("msg")
        pm = _PROFILE_RE.search(msg)
        if not pm:
            non_profile_errors += 1
            continue
        if pm.group("profile") != "std::init":
            # A different profile (e.g. a test:: profile); not counted here.
            continue
        path = m.group("path")
        rule = classify_rule(msg)
        lib = library_for_path(path, boost_root, cache)
        by_rule[rule] += 1
        by_library[lib] += 1
        matrix[lib][rule] += 1
        by_file[path] += 1
        files.add(path)

    total = sum(by_rule.values())
    return {
        "total_violations": total,
        "files": len(files),
        "libraries": len(by_library),
        "crash_markers": crash_markers,
        "non_profile_errors": non_profile_errors,
        "by_rule": {r: by_rule.get(r, 0) for r in RULE_ORDER if by_rule.get(r)},
        "by_library": dict(by_library.most_common()),
        "matrix": {lib: dict(counts) for lib, counts in matrix.items()},
        "top_files": by_file.most_common(50),
    }


# --------------------------------------------------------------------------
# Markdown (step summary)
# --------------------------------------------------------------------------
def _bar(count, maximum, width=24):
    if maximum <= 0:
        return ""
    n = max(1, round(count / maximum * width))
    return "█" * n


def render_markdown(data, top_libraries=20, matrix_libraries=15):
    out = []
    out.append("## std::init profile violations — Boost build\n")
    total = data["total_violations"]
    if total == 0:
        out.append("No `std::init` profile violations found in the log.\n")
        if data["non_profile_errors"]:
            out.append(f"\n_({data['non_profile_errors']} non-profile "
                       "errors seen — likely unrelated Boost/clang issues.)_\n")
        if data["crash_markers"]:
            out.append(f"\n**{data['crash_markers']} clang crash "
                       "marker(s) detected.**\n")
        return "".join(out)

    out.append(
        f"**{total}** violations across **{data['files']}** files in "
        f"**{data['libraries']}** libraries.  \n"
    )
    extra = []
    if data["crash_markers"]:
        extra.append(f"**{data['crash_markers']} clang crash "
                     "marker(s)**")
    if data["non_profile_errors"]:
        extra.append(f"{data['non_profile_errors']} non-profile errors")
    if extra:
        out.append(" · ".join(extra) + "\n")

    # Rule distribution (mermaid pie).
    out.append("\n### By rule kind\n\n")
    out.append("```mermaid\npie showData title Violations by rule\n")
    for rule, count in sorted(data["by_rule"].items(),
                              key=lambda kv: kv[1], reverse=True):
        out.append(f'  "{rule}" : {count}\n')
    out.append("```\n")

    # Top libraries.
    libs = list(data["by_library"].items())[:top_libraries]
    if libs:
        maximum = libs[0][1]
        out.append("\n### Top libraries\n\n")
        out.append("| Library | Violations | |\n|---|--:|:--|\n")
        for lib, count in libs:
            out.append(f"| `{lib}` | {count} | {_bar(count, maximum)} |\n")

    # Rule x library matrix (top libraries as rows, occurring rules as columns).
    present_rules = [r for r in RULE_ORDER if data["by_rule"].get(r)]
    mlibs = list(data["by_library"].items())[:matrix_libraries]
    if mlibs and present_rules:
        out.append(f"\n### Rule × library (top {len(mlibs)})\n\n")
        header = "| Library | " + " | ".join(present_rules) + " | Total |\n"
        sep = "|---|" + "|".join(["--:"] * (len(present_rules) + 1)) + "|\n"
        out.append(header)
        out.append(sep)
        for lib, total_lib in mlibs:
            row = data["matrix"].get(lib, {})
            cells = " | ".join(str(row.get(r, 0)) for r in present_rules)
            out.append(f"| `{lib}` | {cells} | {total_lib} |\n")

    return "".join(out)


# --------------------------------------------------------------------------
# HTML dashboard (self-contained; no external resources)
# --------------------------------------------------------------------------
_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>std::init violations — Boost</title>
<style>
:root { color-scheme: light dark; --fg:#1a1a1a; --bg:#ffffff; --muted:#6b7280;
  --line:#e5e7eb; --bar:#4f46e5; --bar-bg:#eef2ff; --card:#f9fafb; }
@media (prefers-color-scheme: dark) {
  :root { --fg:#e5e7eb; --bg:#0f1115; --muted:#9ca3af; --line:#272b33;
    --bar:#818cf8; --bar-bg:#1e2130; --card:#161922; } }
* { box-sizing: border-box; }
body { margin:0; padding:24px; font:15px/1.5 system-ui,-apple-system,Segoe UI,
  Roboto,sans-serif; color:var(--fg); background:var(--bg); }
h1 { font-size:20px; margin:0 0 4px; } h2 { font-size:15px; margin:28px 0 10px; }
.sub { color:var(--muted); margin:0 0 20px; }
.cards { display:flex; flex-wrap:wrap; gap:12px; margin-bottom:8px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px;
  padding:12px 16px; min-width:120px; }
.card .n { font-size:24px; font-weight:600; } .card .l { color:var(--muted);
  font-size:12px; text-transform:uppercase; letter-spacing:.04em; }
.warn { color:#b91c1c; } @media (prefers-color-scheme: dark){ .warn{color:#fca5a5;} }
table { border-collapse:collapse; width:100%; margin-top:6px; }
th,td { text-align:left; padding:6px 10px; border-bottom:1px solid var(--line);
  white-space:nowrap; } td.n,th.n { text-align:right; }
th { cursor:pointer; user-select:none; position:sticky; top:0; background:var(--bg); }
th:hover { color:var(--bar); }
.barcell { width:40%; } .bar { height:12px; background:var(--bar);
  border-radius:3px; min-width:2px; } .bar-wrap { background:var(--bar-bg);
  border-radius:3px; }
.wrap { overflow-x:auto; } input { padding:6px 10px; border:1px solid var(--line);
  border-radius:8px; background:var(--bg); color:var(--fg); margin-bottom:8px;
  width:240px; max-width:100%; } code { font-family:ui-monospace,monospace; }
</style>
</head>
<body>
<h1>std::init profile violations — Boost</h1>
<p class="sub" id="sub"></p>
<div class="cards" id="cards"></div>

<h2>By rule kind</h2>
<div class="wrap"><table id="rules"><thead><tr>
  <th data-k="rule">Rule</th><th class="n" data-k="count">Count</th>
  <th class="barcell">Share</th></tr></thead><tbody></tbody></table></div>

<h2>By library <input id="libfilter" placeholder="filter libraries…"></h2>
<div class="wrap"><table id="libs"><thead><tr>
  <th data-k="lib">Library</th><th class="n" data-k="count">Violations</th>
  <th class="barcell">Share</th></tr></thead><tbody></tbody></table></div>

<h2>Rule × library</h2>
<div class="wrap"><table id="matrix"><thead></thead><tbody></tbody></table></div>

<script id="data" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById("data").textContent);
const RULE_ORDER = __RULE_ORDER__;
const esc = s => String(s);

document.getElementById("sub").textContent =
  `${D.total_violations} violations · ${D.files} files · ${D.libraries} libraries`;

const cards = [
  ["Violations", D.total_violations], ["Files", D.files],
  ["Libraries", D.libraries], ["Crash markers", D.crash_markers],
  ["Non-profile errors", D.non_profile_errors],
];
document.getElementById("cards").innerHTML = cards.map(([l,n]) =>
  `<div class="card"><div class="n ${l==='Crash markers'&&n>0?'warn':''}">${n}</div>`
  + `<div class="l">${l}</div></div>`).join("");

function bar(count, max) {
  const pct = max > 0 ? Math.max(2, Math.round(count/max*100)) : 0;
  return `<div class="bar-wrap"><div class="bar" style="width:${pct}%"></div></div>`;
}

function renderRows(tbodySel, rows, max, cols) {
  const tb = document.querySelector(tbodySel + " tbody");
  tb.innerHTML = rows.map(r =>
    "<tr>" + cols(r) + `<td class="barcell">${bar(r.count, max)}</td></tr>`
  ).join("");
}

// Rules table
const ruleRows = RULE_ORDER.filter(r => D.by_rule[r])
  .map(r => ({rule:r, count:D.by_rule[r]}));
let ruleMax = Math.max(1, ...ruleRows.map(r => r.count));
renderRows("#rules", ruleRows.sort((a,b)=>b.count-a.count), ruleMax,
  r => `<td><code>${esc(r.rule)}</code></td><td class="n">${r.count}</td>`);

// Libraries table (sortable + filterable)
let libRows = Object.entries(D.by_library).map(([lib,count])=>({lib,count}));
let libMax = Math.max(1, ...libRows.map(r=>r.count));
let libSort = {k:"count", dir:-1};
function drawLibs() {
  const f = document.getElementById("libfilter").value.toLowerCase();
  let rows = libRows.filter(r => r.lib.toLowerCase().includes(f));
  rows.sort((a,b)=> (a[libSort.k] > b[libSort.k] ? 1 : -1) * libSort.dir);
  renderRows("#libs", rows, libMax,
    r => `<td><code>${esc(r.lib)}</code></td><td class="n">${r.count}</td>`);
}
document.querySelectorAll("#libs th[data-k]").forEach(th =>
  th.onclick = () => { const k = th.dataset.k;
    libSort = {k, dir: libSort.k===k ? -libSort.dir : (k==="count"?-1:1)}; drawLibs(); });
document.getElementById("libfilter").oninput = drawLibs;
drawLibs();

// Rule x library matrix
const presentRules = RULE_ORDER.filter(r => D.by_rule[r]);
const mlibs = Object.entries(D.by_library);
const mhead = document.querySelector("#matrix thead");
mhead.innerHTML = "<tr><th>Library</th>" +
  presentRules.map(r=>`<th class="n">${esc(r)}</th>`).join("") +
  "<th class=\\"n\\">Total</th></tr>";
document.querySelector("#matrix tbody").innerHTML = mlibs.map(([lib,total]) => {
  const row = D.matrix[lib] || {};
  return `<tr><td><code>${esc(lib)}</code></td>` +
    presentRules.map(r=>`<td class="n">${row[r]||0}</td>`).join("") +
    `<td class="n">${total}</td></tr>`;
}).join("");
</script>
</body>
</html>
"""


def render_html(data):
    payload = json.dumps(data)
    # The JSON is embedded in a <script> block; neutralize any "</" that could
    # close it early.
    payload = payload.replace("</", "<\\/")
    return (_HTML_TEMPLATE
            .replace("__DATA__", payload)
            .replace("__RULE_ORDER__", json.dumps(RULE_ORDER)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", required=True, help="path to the Boost build log")
    ap.add_argument("--boost-root", default=None,
                    help="Boost checkout root, for precise library attribution "
                         "via header symlink resolution")
    ap.add_argument("--json", default=None, help="write aggregates to this file")
    ap.add_argument("--html", default=None,
                    help="write a self-contained HTML dashboard to this file")
    args = ap.parse_args(argv)

    if not os.path.exists(args.log):
        sys.stderr.write(f"log not found: {args.log}\n")
        # Still emit an empty summary so the workflow step succeeds.
        data = parse_log([], args.boost_root)
    else:
        with open(args.log, "r", errors="replace") as f:
            data = parse_log(f, args.boost_root)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(data, f, indent=2)
    if args.html:
        with open(args.html, "w") as f:
            f.write(render_html(data))

    sys.stdout.write(render_markdown(data))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

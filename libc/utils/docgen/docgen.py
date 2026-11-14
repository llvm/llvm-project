#!/usr/bin/env python3
# ===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===##
from argparse import ArgumentParser, Namespace
import html
import os
from pathlib import Path
import re
import sys
from typing import Callable, Dict, List, Optional

from header import Header

_CPPREFERENCE_BASE = "https://en.cppreference.com/w/c"


class DocgenAPIFormatError(Exception):
    """Raised on fatal formatting errors with a description of a formatting error"""


def check_api(header: Header, api: Dict):
    """
    Checks that docgen yaml files are properly formatted. If there are any
    fatal formatting errors, raises exceptions with error messages useful for
    fixing formatting. Warnings are printed to stderr on non-fatal formatting
    errors. The code that runs after ``check_api(api)`` is called expects that
    ``check_api`` executed without raising formatting exceptions so the yaml
    matches the formatting specified here.

    The yaml file may contain:
    * an optional macros object
    * an optional functions object

    Formatting of ``macros`` and ``functions`` objects
    ==================================================

    If a macros or functions object is present, then it may contain nested
    objects. Each of these nested objects should have a name matching a macro
    or function's name, and each nested object must have the property:
    ``"c-definition"`` or ``"posix-definition"``.

    Description of properties
    =========================
    The defined property is intended to be a reference to a part of the
    standard that defines the function or macro. For the ``"c-definition"`` property,
    this should be a C standard section number. For the ``"posix-definition"`` property,
    this should be a link to the definition.

    :param api: docgen yaml file contents parsed into a dict
    """
    errors = []
    for key in (
        "c-reference-root",
        "standard",
        "posix",
        "family-note",
    ):
        if key in api and not isinstance(api[key], str):
            errors.append(f"error: {key} must be a string")
    if "family-columns" in api:
        cols = api["family-columns"]
        if not isinstance(cols, list) or not all(
            isinstance(c, dict)
            and isinstance(c.get("key"), str)
            and isinstance(c.get("label"), str)
            for c in cols
        ):
            errors.append(
                "error: family-columns must be a list of {key, label} mappings"
            )
    # We require entries to have at least one of these.
    possible_keys = [
        "c-definition",
        "c-error",
        "in-latest-posix",
        "removed-in-posix-2008",
        "removed-in-posix-2024",
        "extension",
    ]

    if "macros" in api and (
        not header.macro_file_exists()
        and not any(header.macro_status(m) != "missing" for m in api["macros"])
    ):
        print(
            f"warning: Macro definitions are listed for {header.name}, "
            "but no macro file can be found in the directory tree rooted "
            f"at {header.macros_dir}. All macros will be listed as not "
            "implemented.",
            file=sys.stderr,
        )

    if "functions" in api and (
        not header.fns_dir_exists()
        and not any(header.fn_status(f) != "missing" for f in api["functions"])
    ):
        print(
            f"warning: Function definitions are listed for {header.name}, "
            "but no function implementation directory exists at "
            f"{header.fns_dir}. All functions will be listed as not "
            "implemented.",
            file=sys.stderr,
        )

    for section, label in (
        ("macros", "Macro"),
        ("functions", "function"),
        ("types", "type"),
        ("objects", "object"),
        ("families", "family"),
    ):
        if section not in api:
            continue
        entries = api[section]
        if not isinstance(entries, dict):
            errors.append(f"error: {section} must be a dict")
            continue
        for name, obj in entries.items():
            if not isinstance(obj, dict) or not any(k in obj for k in possible_keys):
                errors.append(
                    f"error: {label} {name} does not contain at least one "
                    f"required property: {possible_keys}"
                )
                continue
            if "c-reference" in obj and not isinstance(obj["c-reference"], str):
                errors.append(f"error: {label} {name} c-reference must be a string")
            if section == "families" and not isinstance(obj.get("variants"), dict):
                errors.append(f"error: family {name} variants must be a dict")

    if errors:
        raise DocgenAPIFormatError("\n".join(errors))


def _status_badge_html(status: str, accuracy: str = "", symbol: str = "") -> str:
    prefix = f"{html.escape(symbol, quote=True)}: " if symbol else ""
    if status == "implemented":
        if accuracy:
            esc_acc = html.escape(accuracy)
            return (
                f'<span class="hc-status-ulp" title="{prefix}Implemented ({esc_acc})">'
                f"{esc_acc}</span>"
            )
        return (
            f'<span class="hc-status-yes" title="{prefix}Implemented">'
            '<span aria-hidden="true">&#10003;</span>'
            '<span class="hc-sr-only">Implemented</span></span>'
        )
    if status == "experimental":
        return (
            f'<span class="hc-status-exp" title="{prefix}Experimental '
            '(requires -DLLVM_LIBC_ENABLE_EXPERIMENTAL_ENTRYPOINTS=ON)">'
            "Experimental</span>"
        )
    return (
        f'<span class="hc-status-no" title="{prefix}Not Implemented">'
        '<span aria-hidden="true">&ndash;</span>'
        '<span class="hc-sr-only">Not Implemented</span></span>'
    )


def cppreference_url(
    header: Header,
    symbol_name: str = "",
    kind: str = "function",
    item: Optional[Dict] = None,
    api: Optional[Dict] = None,
) -> str:
    if item and "c-reference" in item:
        ref = item["c-reference"]
        return f"{_CPPREFERENCE_BASE}/{ref}" if ref else ""
    root = (api if api is not None else header.api).get("c-reference-root")
    if not root:
        return ""
    if (
        not symbol_name
        or symbol_name.startswith("__")
        or kind not in ("function", "family")
    ):
        return f"{_CPPREFERENCE_BASE}/{root}"
    return f"{_CPPREFERENCE_BASE}/{root}/{symbol_name}"


def posix_url(header: Header, symbol_name: str, kind: str, item: Dict) -> str:
    subpath = (
        f"functions/{symbol_name}.html"
        if kind in ("function", "family") and symbol_name
        else f"basedefs/{header.name.replace('/', '_')}.html"
    )
    if "in-latest-posix" in item:
        return f"https://pubs.opengroup.org/onlinepubs/9799919799/{subpath}"
    if "removed-in-posix-2008" in item:
        return f"https://pubs.opengroup.org/onlinepubs/007904875/{subpath}"
    if "removed-in-posix-2024" in item:
        return f"https://pubs.opengroup.org/onlinepubs/9699919799.2018edition/{subpath}"
    return ""


_C_SECTION_RE = re.compile(r"^([0-9]+|[FGHJK])\.")


def _c_badge_or_link(sec: str, url: str, title_prefix: str) -> str:
    label = html.escape(str(sec))
    if _C_SECTION_RE.match(str(sec)) and url:
        href = html.escape(url, quote=True)
        return (
            f'<a class="hc-c-link" href="{href}" '
            f'title="{title_prefix} {label} on cppreference.com">'
            f'{label} <span aria-hidden="true">&#8599;</span></a>'
        )
    return f'<span class="hc-sec-badge">{label}</span>'


def _c_section_html(
    header: Header,
    name: str,
    kind: str,
    item: Dict,
    api: Optional[Dict] = None,
) -> str:
    sec = item.get("c-definition")
    err = item.get("c-error")
    if not sec and not err:
        return '<span class="hc-status-no" aria-label="Not in C23">&ndash;</span>'
    url = cppreference_url(header, name, kind, item, api)
    parts = []
    if sec:
        parts.append(_c_badge_or_link(str(sec), url, "C23 section"))
    if err:
        parts.append(_c_badge_or_link(str(err), url, "C23 error handling"))
    return '<span class="hc-c-cell">' + " ".join(parts) + "</span>"


def _posix_link_html(header: Header, name: str, kind: str, item: Dict) -> str:
    url = posix_url(header, name, kind, item)
    if not url:
        return '<span class="hc-status-no" aria-label="Not in POSIX">&ndash;</span>'
    if "removed-in-posix-2008" in item:
        label = "Removed (2008)"
    elif "removed-in-posix-2024" in item:
        label = "Removed (2024)"
    else:
        label = "POSIX.1-2024"
    href = html.escape(url, quote=True)
    return (
        f'<a class="hc-posix-link" href="{href}">'
        f'{label} <span aria-hidden="true">&#8599;</span></a>'
    )


def print_symbol_table_md(
    header: Header,
    section_title: str,
    col_label: str,
    items: Dict,
    status_fn: Callable[[str], str],
    kind: str,
    api: Optional[Dict] = None,
):
    print(
        f"## {section_title}\n\n```{{raw}} html\n"
        '<div class="hc-table-wrap"><table class="hc-table">\n'
        f"  <thead><tr><th>{col_label}</th>"
        '<th class="hc-center">Status</th>'
        '<th class="hc-center">C23 Section</th>'
        '<th class="hc-center">POSIX Docs</th></tr></thead>\n'
        "  <tbody>"
    )
    for name in sorted(items.keys()):
        obj = items[name] or {}
        row_id = html.escape(f"{kind}-{name.replace(' ', '-')}", quote=True)
        status = _status_badge_html(status_fn(name))
        c_sec = _c_section_html(header, name, kind, obj, api)
        posix = _posix_link_html(header, name, kind, obj)
        print(
            f'    <tr id="{row_id}">'
            f'<td><code class="hc-sym">{html.escape(name)}</code></td>'
            f'<td class="hc-center">{status}</td>'
            f'<td class="hc-center">{c_sec}</td>'
            f'<td class="hc-center">{posix}</td></tr>'
        )
    print("  </tbody>\n</table></div>\n```\n")


def print_families_md(header: Header, api: Dict):
    families = api["families"]
    cols = api.get("family-columns") or [
        {"key": "float", "label": "float"},
        {"key": "double", "label": "double"},
        {"key": "long_double", "label": "long double"},
        {"key": "float16", "label": "_Float16"},
        {"key": "float128", "label": "_Float128"},
    ]
    groups: Dict[str, Dict] = {}
    for fname, fdata in families.items():
        grp = (fdata or {}).get("group") or "Function Families"
        groups.setdefault(grp, {})[fname] = fdata or {}

    note = api.get("family-note")
    if note:
        print(f"{note}\n")

    col_headers = "".join(
        f'<th class="hc-center">{html.escape(c["label"])}</th>' for c in cols
    )
    for grp_title, grp_items in groups.items():
        print(
            f"## {grp_title}\n\n```{{raw}} html\n"
            '<div class="hc-table-wrap"><table class="hc-table hc-table-families">\n'
            f"  <thead><tr><th>Family</th>{col_headers}"
            '<th class="hc-center">C23 Section</th>'
            '<th class="hc-center">POSIX Docs</th></tr></thead>\n'
            "  <tbody>"
        )
        for fname in sorted(grp_items.keys()):
            fdata = grp_items[fname]
            eval_res = header.evaluate_family(fname, fdata)
            vars_res = eval_res["variants"]
            row_id = html.escape(f"family-{fname}", quote=True)
            cells = []
            for col in cols:
                v_info = vars_res.get(col["key"])
                if v_info is None:
                    cells.append(
                        '<td class="hc-center">'
                        '<span class="hc-status-na" title="Not applicable">N/A</span>'
                        "</td>"
                    )
                else:
                    badge = _status_badge_html(
                        v_info["status"],
                        v_info["accuracy"],
                        v_info["symbol"],
                    )
                    cells.append(f'<td class="hc-center">{badge}</td>')
            c_sec = _c_section_html(header, fname, "family", fdata, api)
            posix = _posix_link_html(header, fname, "family", fdata)
            cells_str = "".join(cells)
            print(
                f'    <tr id="{row_id}">'
                f'<td><code class="hc-sym">{html.escape(fname)}</code></td>'
                f"{cells_str}"
                f'<td class="hc-center">{c_sec}</td>'
                f'<td class="hc-center">{posix}</td></tr>'
            )
        print("  </tbody>\n</table></div>\n```\n")


def _tone(pct: float) -> str:
    if pct >= 99.95:
        return "green"
    return "blue" if pct >= 50.0 else ("amber" if pct > 0.0 else "muted")


def _progress_bar_html(
    pct: float, label: str, tone: str, extra_cls: str = "", exp_pct: float = 0.0
) -> str:
    cls = f"hc-progress-track {extra_cls}".strip()
    esc = html.escape(label, quote=True)
    exp_seg = (
        f'<div class="hc-progress-fill hc-progress-exp" style="width:{exp_pct:.1f}%">'
        "</div>"
        if exp_pct > 0
        else ""
    )
    return (
        f'<div class="{cls}" role="progressbar" aria-label="{esc}" '
        f'aria-valuenow="{pct:.1f}" aria-valuemin="0" aria-valuemax="100">'
        f'<div class="hc-progress-fill" '
        f'style="background:var(--hc-{tone}-text);width:{pct:.1f}%"></div>'
        f"{exp_seg}</div>"
    )


def _mini_card_html(
    scope: str, title: str, imp: int, tot: int, tone: str = "", exp: int = 0
) -> str:
    pct = (imp / tot * 100.0) if tot else 0.0
    exp_pct = (exp / tot * 100.0) if tot else 0.0
    t = tone or _tone(pct)
    bar = _progress_bar_html(pct, f"{scope} {title} completion", t, exp_pct=exp_pct)
    exp_note = f' <span class="hc-exp-note">+{exp:,} exp</span>' if exp else ""
    return (
        f'<div class="hc-mini-card"><div class="hc-mini-title">'
        f"{html.escape(title)}</div>"
        f'<div class="hc-mini-val">{imp:,} / {tot:,} '
        f'<span style="color:var(--hc-{t}-text)">({pct:.1f}%)</span>{exp_note}</div>'
        f"{bar}</div>"
    )


def _precision_grid_html(header_name: str, precisions: List[Dict]) -> str:
    cards = "".join(
        _mini_card_html(
            header_name,
            p["label"],
            p["implemented"],
            p["total"],
            exp=p.get("experimental", 0),
        )
        for p in precisions
    )
    return f'<div class="hc-grid-prec">{cards}</div>'


def _header_pills_html(stats: Dict, short: bool = False) -> str:
    pills = []
    if stats["standard"] == "c23":
        lbl = "C23" if short else "C23 Standard"
        pills.append(f'<span class="hc-pill hc-pill-c23">{lbl}</span>')
    elif stats["standard"] == "iso-tr-18037":
        lbl = "TR 18037" if short else "ISO/IEC TR 18037"
        pills.append(f'<span class="hc-pill hc-pill-ext">{lbl}</span>')
    if stats["posix"] == "posix-2024":
        lbl = "POSIX" if short else "POSIX.1-2024"
        pills.append(f'<span class="hc-pill hc-pill-posix">{lbl}</span>')
    if not pills:
        if stats["removed_posix"]:
            lbl = "Removed" if short else "Removed in POSIX.1-2024"
        else:
            lbl = "Ext" if short else "Extension Header"
        pills.append(f'<span class="hc-pill hc-pill-ext">{lbl}</span>')
    if stats.get("experimental_header"):
        lbl = "Exp" if short else "Experimental"
        pills.append(
            '<span class="hc-pill hc-pill-exp" '
            'title="Requires -DLLVM_LIBC_ENABLE_EXPERIMENTAL_ENTRYPOINTS=ON">'
            f"{lbl}</span>"
        )
    return "".join(pills)


def print_summary_card(header: Header, api: Dict):
    stats = header.get_summary_stats(api)
    ov = stats["overall"]
    imp, tot, exp = ov["implemented"], ov["total"], ov["experimental"]
    if not tot:
        return
    pct = imp / tot * 100.0
    exp_pct = exp / tot * 100.0
    tone = _tone(pct)
    pills_html = _header_pills_html(stats)

    cards = "".join(
        _mini_card_html(
            header.name,
            title,
            stats[k]["implemented"],
            stats[k]["total"],
            c,
            stats[k]["experimental"],
        )
        for k, title, c in (
            ("c23", "C23 Standard", "green"),
            ("tr18037", "ISO/IEC TR 18037", "purple"),
            ("posix_spec", "POSIX.1-2024", "blue"),
            ("families", "Base Families", "green"),
            ("raw_functions", "Functions", "green"),
            ("macros", "Macros", "blue"),
            ("types", "Types", "purple"),
            ("objects", "Global Variables", "amber"),
        )
        if stats[k]["total"]
    )
    overall_bar = _progress_bar_html(
        pct,
        f"{header.name} overall implementation progress",
        tone,
        "hc-progress-lg",
        exp_pct,
    )
    exp_note = (
        f' <span class="hc-exp-note">(+{exp:,} experimental)</span>' if exp else ""
    )
    print(
        "```{raw} html\n"
        f'<div class="hc-card"><div class="hc-summary-top">'
        f'<div class="hc-pill-row">{pills_html}</div>'
        f'<div class="hc-overall-stat">{imp:,} / {tot:,} '
        f'<span style="color:var(--hc-{tone}-text)">({pct:.1f}%)</span>'
        f"{exp_note}</div></div>"
        f'{overall_bar}<div class="hc-grid-cards">{cards}</div></div>'
    )
    if stats["precisions"]:
        print(
            '<div class="hc-card">'
            '<div class="hc-subcard-title">Precision Coverage</div>'
            f'{_precision_grid_html(header.name, stats["precisions"])}</div>'
        )
    print("```\n")


def print_header_status_md(header: Header, api: Dict):
    print(f"# {header.name}\n")
    print_summary_card(header, api)

    if api.get("families"):
        print_families_md(header, api)

    if api.get("types"):
        print_symbol_table_md(
            header, "Types", "Type", api["types"], header.type_status, "type", api
        )

    if api.get("objects"):
        print_symbol_table_md(
            header,
            "Global Variables",
            "Global Variable",
            api["objects"],
            header.object_status,
            "object",
            api,
        )

    if api.get("macros"):
        print_symbol_table_md(
            header, "Macros", "Macro", api["macros"], header.macro_status, "macro", api
        )

    if api.get("functions"):
        print_symbol_table_md(
            header,
            "Functions",
            "Function",
            api["functions"],
            header.fn_status,
            "function",
            api,
        )


# This code implicitly relies on docgen.py being in the same dir as the yaml
# files and is likely to need to be fixed when re-integrating docgen into
# hdrgen.
def get_choices() -> list:
    choices = []
    for path in Path(__file__).parent.rglob("*.yaml"):
        fname = path.with_suffix(".h").name
        if path.parent != Path(__file__).parent:
            fname = path.parent.name + os.sep + fname
        choices.append(fname)
    return sorted(choices)


def _std_card_html(title: str, imp: int, tot: int, exp: int, tone: str) -> str:
    pct = (imp / tot * 100.0) if tot else 0.0
    exp_pct = (exp / tot * 100.0) if tot else 0.0
    bar = _progress_bar_html(pct, title, tone, "hc-progress-lg", exp_pct)
    exp_note = (
        f' <span class="hc-exp-note">(+{exp:,} experimental)</span>' if exp else ""
    )
    return (
        f'<div class="hc-card"><div class="hc-std-title">{title}</div>'
        f'<div class="hc-std-val" style="color:var(--hc-{tone}-text)">'
        f"{pct:.1f}% <span>({imp:,} / {tot:,})</span>{exp_note}</div>{bar}</div>"
    )


def _chip_html(
    header: Header, api: Dict, name: str, kind: str, item: Dict, status: str
) -> str:
    cls_map = {
        "implemented": "hc-chip-impl",
        "experimental": "hc-chip-exp",
        "missing": "hc-chip-missing",
    }
    cls = cls_map[status]
    esc = html.escape(name)
    cdef = str(item.get("c-definition") or "")
    url, title = "", ""
    if api.get("standard") == "c23" and _C_SECTION_RE.match(cdef):
        url = cppreference_url(header, name, kind, item, api)
        title = f"{name} on cppreference.com"
    if not url:
        url = posix_url(header, name, kind, item)
        title = f"{name} in POSIX"
    if status == "experimental":
        suffix = (
            " (Experimental: requires -DLLVM_LIBC_ENABLE_EXPERIMENTAL_ENTRYPOINTS=ON)"
        )
        title = (title or name) + suffix
    if url:
        href = html.escape(url, quote=True)
        esc_t = html.escape(title, quote=True)
        return (
            f'<a class="{cls}" href="{href}" target="_blank" rel="noopener" '
            f'title="{esc_t}">{esc}</a>'
        )
    if title:
        return (
            f'<span class="{cls}" title="{html.escape(title, quote=True)}">{esc}</span>'
        )
    return f'<span class="{cls}">{esc}</span>'


def _chip_group_html(
    header: Header, api: Dict, entries: List, label: str, tone: str, status: str
) -> str:
    if not entries:
        return ""
    chips = "".join(
        _chip_html(header, api, n, k, it, status) for n, k, it in entries[:16]
    )
    rem = len(entries) - 16
    more = f'<span class="hc-chip-more">+{rem} more...</span>' if rem > 0 else ""
    return (
        f'<div class="hc-chip-row"><span class="hc-chip-lbl" '
        f'style="color:var(--hc-{tone}-text)">{label} ({len(entries)}):</span>'
        f"{chips}{more}</div>"
    )


def _toolbar_html(total_headers: int) -> str:
    std_btns = "".join(
        f'<button type="button" class="filter-btn hc-toggle-btn" '
        f'aria-pressed="{"true" if v == "all" else "false"}" '
        f"onclick=\"setStandardFilter('{v}', this)\">{lbl}</button>"
        for v, lbl in (
            ("all", "All Headers"),
            ("c23", "C23 Standard"),
            ("posix", "POSIX.1-2024"),
            ("ext", "Extensions"),
        )
    )
    st_btns = "".join(
        f'<button type="button" class="status-btn hc-toggle-btn" '
        f'aria-pressed="{"true" if v == "all" else "false"}" '
        f"onclick=\"setStatusFilter('{v}', this)\">{lbl}</button>"
        for v, lbl in (
            ("all", "All"),
            ("complete", "Complete (100%)"),
            ("inprogress", "In Progress (1\u201399%)"),
            ("experimental", "Experimental"),
            ("planned", "Planned (0%)"),
        )
    )
    return (
        '<div class="dashboard-controls" role="region" '
        'aria-label="Filter and sort headers"><div class="hc-ctrl-row">'
        '<div class="hc-search-wrap"><label for="header-search" class="hc-sr-only">'
        "Search headers or symbols</label>"
        '<input type="search" id="header-search" class="hc-input" '
        'aria-label="Search headers or symbols" '
        'placeholder="Search headers (stdio, sys/stat) or symbols (printf, errno)..." '
        'oninput="filterHeaders()"></div>'
        '<div class="hc-ctrl-grp"><label for="sort-select" class="hc-ctrl-lbl">'
        'Sort:</label><select id="sort-select" class="hc-input" '
        'onchange="sortHeaders()">'
        '<option value="name-asc">Header Name (A\u2013Z)</option>'
        '<option value="pct-desc">Completion % (High \u2192 Low)</option>'
        '<option value="pct-asc">Completion % (Low \u2192 High)</option>'
        '<option value="missing-desc">Most Missing Items</option></select></div>'
        '<div class="hc-ctrl-grp">'
        '<button type="button" class="hc-btn" onclick="toggleAllDetails(true)">'
        "Expand All</button>"
        '<button type="button" class="hc-btn" onclick="toggleAllDetails(false)">'
        "Collapse All</button></div></div>"
        '<div class="hc-ctrl-row">'
        '<div class="hc-ctrl-grp" role="group" aria-label="Filter by standard">'
        f'<span class="hc-ctrl-lbl">Standard:</span>{std_btns}</div>'
        '<div class="hc-ctrl-grp" role="group" '
        'aria-label="Filter by implementation status">'
        f'<span class="hc-ctrl-lbl">Status:</span>{st_btns}</div></div>'
        '<div id="filter-count" class="hc-filter-count" role="status" '
        'aria-live="polite" aria-atomic="true">'
        f"Showing {total_headers} of {total_headers} headers</div></div>"
    )


def generate_index_md():
    headers_data = []
    totals = {
        k: [0, 0, 0]
        for k in ("c23", "posix_spec", "functions", "macros", "types", "objects")
    }
    for hname in get_choices():
        hdr = Header(hname)
        check_api(hdr, hdr.api)
        stats = hdr.get_summary_stats(hdr.api)
        headers_data.append((hdr, hdr.api, stats))
        for k, triple in totals.items():
            triple[0] += stats[k]["implemented"]
            triple[1] += stats[k]["total"]
            triple[2] += stats[k]["experimental"]

    n_c23 = sum(1 for _, _, s in headers_data if s["standard"] == "c23")
    n_posix = sum(1 for _, _, s in headers_data if s["posix"] == "posix-2024")
    c23_card = _std_card_html("C23 Standard Completeness", *totals["c23"], "green")
    pos_card = _std_card_html(
        "POSIX.1-2024 Completeness", *totals["posix_spec"], "blue"
    )
    cat_cards = "".join(
        _mini_card_html(
            "Overall", title, totals[k][0], totals[k][1], tone, totals[k][2]
        )
        for k, title, tone in (
            ("functions", "Functions", "green"),
            ("macros", "Macros", "blue"),
            ("types", "Types", "purple"),
            ("objects", "Global Variables", "amber"),
        )
    )
    print(
        "(implementation_status)=\n\n(headers_index)=\n\n# Headers\n\n"
        f"Live symbol coverage across {len(headers_data)} headers ({n_c23} C23 "
        f"standard headers, {n_posix} POSIX.1-2024 headers, and extensions), verified "
        "directly against LLVM-libc source files, header generation specifications, "
        "and exported entrypoints.  Symbols gated behind "
        "`LLVM_LIBC_ENABLE_EXPERIMENTAL_ENTRYPOINTS` (off by default) are marked "
        "separately as experimental and excluded from default completion percentages.  "
        "Purely compiler-provided freestanding C headers (`iso646.h`, `stdalign.h`, "
        "`stdarg.h`, `stdatomic.h`, `stdbool.h`, `stddef.h`, `stdnoreturn.h`, and "
        "`tgmath.h`) come from the host compiler (`Clang` or `GCC`) rather than "
        "LLVM-libc and are omitted below.\n\n"
        "```{raw} html\n"
        f'<div class="hc-grid-standards">{c23_card}{pos_card}</div>'
        f'<div class="hc-grid-cards">{cat_cards}</div>\n```\n\n'
        "## All Headers\n\n"
        "Filter, search, or click any header to inspect category breakdowns, "
        "precision coverage, and experimental, missing, or implemented symbols.\n\n"
        "```{raw} html\n"
        f"{_toolbar_html(len(headers_data))}"
        '<div id="headers-list" class="hc-headers-list">'
    )
    for hdr, api, stats in headers_data:
        ov = stats["overall"]
        imp, tot, exp = ov["implemented"], ov["total"], ov["experimental"]
        pct = (imp / tot * 100.0) if tot else 0.0
        exp_pct = (exp / tot * 100.0) if tot else 0.0
        std_tags = []
        if stats["standard"] == "c23":
            std_tags.append("c23")
        if stats["posix"] == "posix-2024":
            std_tags.append("posix")
        if not std_tags:
            std_tags.append("ext")
        status_attr = (
            "complete" if tot and imp == tot else ("inprogress" if imp else "planned")
        )
        syms = [
            s
            for sec in ("families", "functions", "macros", "types", "objects")
            for s in (api.get(sec) or {})
        ] + [
            v
            for fd in (api.get("families") or {}).values()
            for v in ((fd or {}).get("variants") or {}).values()
            if v
        ]
        by_st: Dict[str, List] = {"experimental": [], "missing": [], "implemented": []}
        sym_sources = [
            (
                "family",
                api.get("families") or {},
                lambda n, d: hdr.evaluate_family(n, d)["status"],
            ),
            ("function", api.get("functions") or {}, lambda n, _: hdr.fn_status(n)),
            ("macro", api.get("macros") or {}, lambda n, _: hdr.macro_status(n)),
            ("type", api.get("types") or {}, lambda n, _: hdr.type_status(n)),
            ("object", api.get("objects") or {}, lambda n, _: hdr.object_status(n)),
        ]
        for kind, entries, st_fn in sym_sources:
            for name in sorted(entries.keys()):
                item = entries[name] or {}
                by_st[st_fn(name, item)].append((name, kind, item))

        mini_cards = "".join(
            _mini_card_html(
                hdr.name,
                title,
                stats[k]["implemented"],
                stats[k]["total"],
                c,
                stats[k]["experimental"],
            )
            for k, title, c in (
                ("families", "Base Families", "green"),
                ("raw_functions", "Functions", "green"),
                ("macros", "Macros", "blue"),
                ("types", "Types", "purple"),
                ("objects", "Global Variables", "amber"),
            )
            if stats[k]["total"]
        )
        prec_block = (
            '<div class="hc-drawer-prec">'
            '<div class="hc-subcard-title">Precision Coverage</div>'
            f'{_precision_grid_html(hdr.name, stats["precisions"])}</div>'
            if stats["precisions"]
            else ""
        )
        exp_html = _chip_group_html(
            hdr,
            api,
            by_st["experimental"],
            "Experimental Symbols",
            "amber",
            "experimental",
        )
        miss_html = _chip_group_html(
            hdr, api, by_st["missing"], "Missing Symbols", "muted", "missing"
        )
        impl_html = _chip_group_html(
            hdr,
            api,
            by_st["implemented"],
            "Implemented Symbols",
            "green",
            "implemented",
        )
        stem = html.escape(hdr.stem, quote=True)
        hname = html.escape(hdr.name, quote=True)
        syms_attr = html.escape(" ".join(syms).lower(), quote=True)
        bar = _progress_bar_html(
            pct, f"{hdr.name} implementation progress", _tone(pct), exp_pct=exp_pct
        )
        has_exp = "true" if (exp > 0 or stats.get("experimental_header")) else "false"
        frac_str = f"({imp}/{tot}, +{exp} exp)" if exp else f"({imp}/{tot})"
        print(
            f'<details class="header-card" data-stem="{stem}" data-name="{hname}" '
            f'data-standard="{" ".join(std_tags)}" data-status="{status_attr}" '
            f'data-experimental="{has_exp}" '
            f'data-pct="{pct:.2f}" data-missing="{tot - imp}" '
            f'data-symbols="{syms_attr}"><summary>'
            '<div class="hc-header-title-col">'
            '<span class="hc-chevron" aria-hidden="true">&#9656;</span>'
            f'<code class="hc-sym">&lt;{hname}&gt;</code>'
            f"{_header_pills_html(stats, short=True)}</div>"
            f'<div class="hc-thermometer-col">{bar}'
            f'<span class="hc-row-pct">{pct:.1f}%</span>'
            f'<span class="hc-row-frac">{frac_str}</span></div>'
            f'<div class="hc-action-col"><a class="hc-page-link" href="{stem}.html" '
            f'onclick="event.stopPropagation();" '
            f'aria-label="Open {hname} status page">'
            'Page <span aria-hidden="true">&#8599;</span></a></div></summary>'
            f'<div class="hc-drawer"><div class="hc-grid-cards">{mini_cards}</div>'
            f"{prec_block}{exp_html}{miss_html}{impl_html}"
            f'<div class="hc-drawer-foot"><a class="hc-page-link" href="{stem}.html">'
            f"View full &lt;{hname}&gt; page &rarr;</a></div></div></details>"
        )
    stems = "\n".join(hdr.stem for hdr, _, _ in headers_data)
    print(f"</div>\n```\n\n```{{toctree}}\n:hidden: true\n:maxdepth: 1\n\n{stems}\n```")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("header_name", nargs="?", choices=get_choices())
    group.add_argument("--index", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.index:
        generate_index_md()
    else:
        header = Header(args.header_name)
        check_api(header, header.api)
        print_header_status_md(header, header.api)

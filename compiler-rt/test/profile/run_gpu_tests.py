#!/usr/bin/env python3
"""Minimal lit-style runner for the HIP device-PGO tests.

The compiler-rt profile lit suite (and llvm-lit / FileCheck) is not part of the
installed ROCm artifact, but the toolchain, the amdgcn device profile runtime,
and the HIP runtime are. This runner executes the
``compiler-rt/test/profile/{GPU,AMDGPU}/*.hip`` tests directly against an
installed toolchain on a real GPU runner, interpreting just the slice of lit
markup those tests use:

  - ``// REQUIRES:`` / ``// UNSUPPORTED:`` boolean feature gating,
  - ``// RUN:`` lines (with ``\\`` continuations) and the fixed substitution set
    (%clang, %s, %t[.*], %amdgpu_arch, %hip_lib_path, %run, %%),
  - delegation to ``FileCheck`` / ``not`` (real binaries if present on PATH,
    otherwise shims backed by the ``filecheck`` PyPI package and a tiny
    exit-code inverter).

Each RUN line is executed via ``bash -e -o pipefail -c`` so pipes, redirection
and globbing behave as under lit. A test passes iff all its RUN lines exit 0.
"""

import argparse
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

# --- feature detection ------------------------------------------------------


def _count_visible_gpus(toolchain_bin):
    """Number of GPUs actually visible to the runtime, or 0 if unknown.

    Uses the toolchain's ``amdgpu-arch`` (one line per visible device). Unlike
    the KFD topology under ``/sys/class/kfd`` this reflects what HIP/ROCr really
    exposes -- it honours ``ROCR_VISIBLE_DEVICES`` / ``HIP_VISIBLE_DEVICES`` and
    container device limits, so it matches what a test's ``hipGetDeviceCount``
    will see. It is also portable: Windows has no ``/dev/kfd``, but does ship
    ``amdgpu-arch``.
    """
    if not toolchain_bin:
        return 0
    tb = Path(toolchain_bin)
    exe = next(
        (str(tb / c) for c in ("amdgpu-arch", "amdgpu-arch.exe") if (tb / c).exists()),
        None,
    )
    if exe is None:
        return 0
    try:
        proc = subprocess.run(exe, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return 0
    if proc.returncode != 0:
        return 0
    return sum(1 for line in proc.stdout.splitlines() if line.strip())


def detect_features(toolchain_bin=None, force=None):
    """Return the set of lit features available on this runner.

    hip/amdgpu are assumed present (this runner only ever drives GPU tests on a
    runner that has the toolchain + HIP). ``multi-device`` is derived from the
    number of GPUs the runtime actually exposes (>= 2), via ``amdgpu-arch``.
    """
    features = {"hip", "amdgpu"}
    if sys.platform.startswith("linux"):
        features.add("linux")
    elif sys.platform.startswith("win"):
        features.add("windows")

    if _count_visible_gpus(toolchain_bin) >= 2:
        features.add("multi-device")

    if force:
        for f in force:
            features.add(f)
    return features


# --- boolean expression evaluation (REQUIRES / UNSUPPORTED) ------------------

_TOKEN_RE = re.compile(r"\s*(\(|\)|\|\||&&|!|[\w.+-]+)\s*")


def _clause_to_py(clause):
    out = []
    for tok in _TOKEN_RE.findall(clause):
        if tok == "||":
            out.append(" or ")
        elif tok == "&&":
            out.append(" and ")
        elif tok == "!":
            out.append(" not ")
        elif tok in ("(", ")"):
            out.append(tok)
        elif tok == "true":
            out.append("True")
        elif tok == "false":
            out.append("False")
        else:
            out.append("(%r in FEATURES)" % tok)
    return "".join(out) or "True"


def eval_requires(expr, features):
    """All comma-separated clauses must be true."""
    return all(
        eval(_clause_to_py(c), {"__builtins__": {}}, {"FEATURES": features})
        for c in expr.split(",")
        if c.strip()
    )


def eval_unsupported(expr, features):
    """Unsupported if any comma-separated clause is true."""
    return any(
        eval(_clause_to_py(c), {"__builtins__": {}}, {"FEATURES": features})
        for c in expr.split(",")
        if c.strip()
    )


# --- test parsing -----------------------------------------------------------

_DIRECTIVE_RE = re.compile(r"(?://|#)\s*(RUN|REQUIRES|UNSUPPORTED):\s?(.*)")


def parse_test(path):
    """Return (run_lines, requires, unsupported) for a test file."""
    runs, requires, unsupported = [], [], []
    cont = None
    for raw in Path(path).read_text(errors="replace").splitlines():
        m = _DIRECTIVE_RE.search(raw)
        if cont is not None:
            # Continuation of a previous RUN line.
            text = raw
            cm = re.search(r"(?://|#)\s*RUN:\s?(.*)", raw)
            if cm:
                text = cm.group(1)
            cont += " " + text.strip()
            if cont.rstrip().endswith("\\"):
                cont = cont.rstrip()[:-1]
            else:
                runs.append(cont)
                cont = None
            continue
        if not m:
            continue
        kind, body = m.group(1), m.group(2)
        if kind == "REQUIRES":
            requires.append(body.strip())
        elif kind == "UNSUPPORTED":
            unsupported.append(body.strip())
        elif kind == "RUN":
            if body.rstrip().endswith("\\"):
                cont = body.rstrip()[:-1]
            else:
                runs.append(body)
    return runs, requires, unsupported


# --- substitutions ----------------------------------------------------------


def make_substitutions(clang, clangxx, src, tprefix, arch, hip_lib_path):
    # Order matters: longer / more specific tokens first; %% resolved last.
    return [
        ("%clangxx", clangxx),
        ("%clang", clang),
        ("%amdgpu_arch", arch),
        ("%hip_lib_path", hip_lib_path),
        ("%run ", ""),
        ("%s", str(src)),
        ("%t", tprefix),
        ("%%", "%"),
    ]


def apply_substitutions(line, subs):
    for token, value in subs:
        line = line.replace(token, value)
    return line


# --- tool shims (FileCheck / not) -------------------------------------------


def ensure_tools(toolchain_bin, workdir):
    """Build a PATH that resolves clang/llvm-*, FileCheck and not.

    Prefers real binaries under toolchain_bin; falls back to shims for FileCheck
    (PyPI ``filecheck``) and ``not`` (exit-code inverter).
    """
    shim_dir = workdir / "shims"
    shim_dir.mkdir(parents=True, exist_ok=True)
    path = os.pathsep.join(
        [str(toolchain_bin), str(shim_dir), os.environ.get("PATH", "")]
    )

    def have(tool):
        # File-based check (shutil.which is quirky across OSes / Git Bash). The
        # shims are extensionless bash scripts, which Git Bash resolves via the
        # shebang, so a real binary is anything matching tool or tool.exe.
        tb = Path(toolchain_bin)
        return (tb / tool).exists() or (tb / (tool + ".exe")).exists()

    def write_shim(name, body):
        p = shim_dir / name
        p.write_text(body)
        p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    if not have("FileCheck"):
        write_shim(
            "FileCheck",
            "#!/usr/bin/env bash\n"
            'if command -v filecheck >/dev/null 2>&1; then exec filecheck "$@"; fi\n'
            'exec python3 -m filecheck "$@"\n',
        )
    if not have("not"):
        write_shim(
            "not",
            "#!/usr/bin/env bash\n"
            'if [ "$1" = "--crash" ]; then shift; "$@"; ec=$?; '
            "[ $ec -ge 128 ] && exit 0 || exit 1; fi\n"
            '"$@"; ec=$?; [ $ec -eq 0 ] && exit 1 || exit 0\n',
        )
    return path


# --- execution --------------------------------------------------------------


def run_one(path, args, features, base_env):
    runs, requires, unsupported = parse_test(path)

    for expr in requires:
        if not eval_requires(expr, features):
            return "UNSUPPORTED", "missing requirement: %s" % expr
    for expr in unsupported:
        if eval_unsupported(expr, features):
            return "UNSUPPORTED", "unsupported: %s" % expr
    if not runs:
        return "UNSUPPORTED", "no RUN lines"

    workdir = Path(tempfile.mkdtemp(prefix="profgpu-"))
    tprefix = str(workdir / "t")
    subs = make_substitutions(
        args.clang,
        args.clangxx,
        Path(path).resolve(),
        tprefix,
        args.amdgpu_arch,
        args.hip_lib_path,
    )

    if args.dry_run:
        print("# %s" % path)
        for line in runs:
            print("    " + apply_substitutions(line, subs).strip())
        return "DRYRUN", ""

    env = dict(base_env)
    env["PATH"] = ensure_tools(Path(args.toolchain_bin), workdir)
    timeout = args.timeout if args.timeout and args.timeout > 0 else None
    for line in runs:
        cmd = apply_substitutions(line, subs).strip()
        try:
            proc = subprocess.run(
                ["bash", "-e", "-o", "pipefail", "-c", cmd],
                cwd=str(workdir),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as e:
            out = e.stdout or ""
            err = e.stderr or ""
            if isinstance(out, bytes):
                out = out.decode("utf-8", "replace")
            if isinstance(err, bytes):
                err = err.decode("utf-8", "replace")
            detail = "RUN timed out after %gs: %s\n%s%s" % (
                timeout,
                cmd,
                out,
                err,
            )
            if not args.keep:
                shutil.rmtree(workdir, ignore_errors=True)
            return "FAIL", detail
        if proc.returncode != 0:
            detail = "RUN failed (rc=%d): %s\n%s%s" % (
                proc.returncode,
                cmd,
                proc.stdout,
                proc.stderr,
            )
            if not args.keep:
                shutil.rmtree(workdir, ignore_errors=True)
            return "FAIL", detail
    if not args.keep:
        shutil.rmtree(workdir, ignore_errors=True)
    return "PASS", ""


def discover(paths):
    tests = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            tests.extend(sorted(str(x) for x in p.rglob("*.hip")))
        elif p.is_file():
            tests.append(str(p))
    return tests


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("tests", nargs="+", help="Test files or directories")
    ap.add_argument(
        "--toolchain-bin", required=False, help="Directory with clang and llvm-* tools"
    )
    ap.add_argument("--hip-lib-path", default="", help="Directory with libamdhip64")
    ap.add_argument("--amdgpu-arch", default="native")
    ap.add_argument("--clang", help="Override clang path")
    ap.add_argument("--clangxx", help="Override clang++ path")
    ap.add_argument(
        "--feature",
        action="append",
        default=[],
        help="Force-enable an extra lit feature",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved RUN lines without executing",
    )
    ap.add_argument("--keep", action="store_true", help="Keep per-test temp dirs")
    ap.add_argument(
        "--timeout",
        type=float,
        default=600,
        help="Per-RUN-line timeout in seconds (<=0 disables); "
        "guards against a hung GPU/compiler wedging the run",
    )
    args = ap.parse_args()

    if not args.dry_run and not args.toolchain_bin:
        ap.error("--toolchain-bin is required unless --dry-run is given")

    if args.toolchain_bin:
        binp = Path(args.toolchain_bin)
        args.clang = args.clang or str(binp / "clang")
        args.clangxx = args.clangxx or str(binp / "clang++")
    else:
        args.clang = args.clang or "clang"
        args.clangxx = args.clangxx or "clang++"

    features = detect_features(args.toolchain_bin, args.feature)
    print("# features: %s" % ", ".join(sorted(features)))

    base_env = dict(os.environ)
    if args.toolchain_bin:
        lib_dirs = [
            str(Path(args.toolchain_bin).parent / "lib"),  # toolchain libs
        ]
        if args.hip_lib_path:
            lib_dirs.append(args.hip_lib_path)
        existing = base_env.get("LD_LIBRARY_PATH", "")
        base_env["LD_LIBRARY_PATH"] = os.pathsep.join(
            [d for d in lib_dirs if d] + ([existing] if existing else [])
        )

    tests = discover(args.tests)
    if not tests:
        print("error: no tests found", file=sys.stderr)
        return 2

    results = {"PASS": [], "FAIL": [], "UNSUPPORTED": [], "DRYRUN": []}
    for t in tests:
        status, detail = run_one(t, args, features, base_env)
        results[status].append(t)
        if status == "FAIL":
            print("FAIL: %s" % t)
            print(detail)
        elif status in ("PASS", "UNSUPPORTED"):
            print("%s: %s" % (status, t))

    print(
        "\n# summary: %d passed, %d failed, %d unsupported (of %d)"
        % (
            len(results["PASS"]),
            len(results["FAIL"]),
            len(results["UNSUPPORTED"]),
            len(tests),
        )
    )
    return 1 if results["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())

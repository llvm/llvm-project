#!/usr/bin/env python3
# ===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===##
"""Same-machine A/B of libc++ lock-free vs lock-based atomic<shared_ptr>.

Runs atomic_shared_ptr.bench.cpp and atomic_shared_ptr_lock_based.bench.cpp
separately (identical Google Benchmark names), writes two LNT files, a
normalized ratio table (same-run uint64 CAS), PNG line charts of the same
comparison, then optionally libcxx/utils/compare-benchmarks. Do not publish
that tool's Geomean.

Contended thread counts are not a Google Benchmark CLI flag. The bench
header reads ATOMIC_SP_BENCH_THREADS at registration time. libc++'s
%{exec} (libcxx/utils/run.py) starts the test with a clean environment,
so this script compiles via lit (dry-run) and then runs t.tmp.exe itself.

Requires a configured libc++ build (bootstrap or runtimes). Charts and
compare-benchmarks both need: pip install -r libcxx/utils/requirements.txt
(charts use the matplotlib dependency listed there). Pass --no-chart to
skip charts entirely without needing matplotlib installed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

THREADS_ENV = "ATOMIC_SP_BENCH_THREADS"
DEFAULT_THREADS = ",".join(str(i) for i in range(1, (os.cpu_count() or 1) + 1))
DEFAULT_MIN_TIME = "1s"
DEFAULT_REPETITIONS = 1
MAX_THREADS = 1024
THREADS_RE = re.compile(r"^[1-9][0-9]*(,[1-9][0-9]*)*$")
MIN_TIME_RE = re.compile(r"^[0-9]+([.][0-9]+)?(s|ms)?$")
AGGREGATE_SUFFIXES = ("_mean", "_median", "_stddev", "_cv")
U64_CAS_CONTENDED = "std::atomic<uint64_t>::compare_exchange_strong() (contended)"
SP_CONTENDED_OPS = ("store", "compare_exchange_strong", "load")
SP_UNCONTENDED = (
    "std::atomic<shared_ptr<T>>::load() (uncontended)",
    "std::atomic<shared_ptr<T>>::store() (uncontended)",
    "std::atomic<shared_ptr<T>>::exchange() (uncontended)",
    "std::atomic<shared_ptr<T>>::compare_exchange_strong() (uncontended)",
)

LOCK_FREE_SRC = "atomic_shared_ptr.bench.cpp"
LOCK_BASED_SRC = "atomic_shared_ptr_lock_based.bench.cpp"

LOCK_FREE_COLOR = "#2a78d6"
LOCK_BASED_COLOR = "#e34948"
CHARTS_DIR_NAME = "charts"

LIBCXX_TEST = Path("libcxx") / "test"
BOOTSTRAP_LIBCXX_TEST = Path("runtimes") / "runtimes-bins" / LIBCXX_TEST
BENCHMARKS_DIR = Path("benchmarks")
LIT_OUTPUT_DIR = BENCHMARKS_DIR / "Output"
LIBCXX_UTILS = Path("libcxx") / "utils"
LIBCXX_LIT = LIBCXX_UTILS / "libcxx-lit"
LLVM_LIT = Path("bin") / "llvm-lit"
LIT_BENCH_EXE = "t.tmp.exe"
BENCH_HEADER = BENCHMARKS_DIR / "atomic_shared_ptr_bench.h"
ATOMIC_SP_HEADERS = (
    Path("libcxx") / "include" / "__memory" / "atomic_shared_ptr.h",
    Path("libcxx") / "include" / "__memory" / "atomic_shared_ptr_lock_free.h",
    Path("libcxx") / "include" / "__memory" / "atomic_shared_ptr_lock_based.h",
)


def die(message: str, code: int = 1) -> NoReturn:
    print(f"{Path(sys.argv[0]).name}: error: {message}", file=sys.stderr)
    raise SystemExit(code)


def log(message: str) -> None:
    print(f"{Path(sys.argv[0]).name}: {message}", file=sys.stderr)


def strip_aggregate_suffix(name: str) -> str:
    for suffix in AGGREGATE_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def selected_benchmarks(js: dict[str, Any]) -> list[dict[str, Any]]:
    benches = js["benchmarks"]
    have_median = any(
        bm.get("aggregate_name") == "median" or bm["name"].endswith("_median")
        for bm in benches
    )
    if have_median:
        return [
            bm
            for bm in benches
            if bm.get("aggregate_name") == "median" or bm["name"].endswith("_median")
        ]
    return [bm for bm in benches if bm.get("run_type", "iteration") != "aggregate"]


def benchmarks_for_repetition(
    js: dict[str, Any], repetition_index: int
) -> list[dict[str, Any]]:
    """Raw (non-aggregate) rows belonging to one --benchmark_repetitions run.

    Only present when run_exe_to_lnt was invoked with repetitions > 1, since
    that is the only case where Google Benchmark's JSON carries per-repetition
    rows alongside the aggregate ones.
    """
    return [
        bm
        for bm in js["benchmarks"]
        if bm.get("run_type", "iteration") != "aggregate"
        and bm.get("repetition_index") == repetition_index
    ]


def thread_count(name: str) -> int | None:
    marker = "/threads:"
    if marker not in name:
        return None
    tail = strip_aggregate_suffix(name.rsplit(marker, 1)[1])
    return int(tail)


def times_by_threads(benches: list[dict[str, Any]], prefix: str) -> dict[int, float]:
    out: dict[int, float] = {}
    for bm in benches:
        name = strip_aggregate_suffix(bm["name"])
        if not name.startswith(prefix):
            continue
        threads = thread_count(name)
        if threads is None:
            continue
        out[threads] = float(bm["real_time"])
    return out


def uncontended_time(benches: list[dict[str, Any]], name: str) -> float | None:
    for bm in benches:
        if strip_aggregate_suffix(bm["name"]) == name:
            return float(bm["real_time"])
    return None


def format_ratio_row(values: list[float | None]) -> str:
    cells = []
    for value in values:
        cells.append(f"{value:.1f}" if value is not None else "n/a")
    return " | ".join(cells)


def print_normalized_report(lf_json: Path, lb_json: Path, report_out: Path) -> None:
    """Ratios to same-run same-thread uint64 CAS. Do not publish geomean."""
    lf_js = json.loads(lf_json.read_text(encoding="utf-8"))
    lb_js = json.loads(lb_json.read_text(encoding="utf-8"))
    lf = selected_benchmarks(lf_js)
    lb = selected_benchmarks(lb_js)
    lf_base = times_by_threads(lf, U64_CAS_CONTENDED)
    lb_base = times_by_threads(lb, U64_CAS_CONTENDED)
    threads = sorted(set(lf_base) & set(lb_base))
    if not threads:
        die("normalized report: no overlapping uint64 CAS contended thread counts")

    lines: list[str] = []
    lf_load = lf_js.get("context", {}).get("load_avg")
    lb_load = lb_js.get("context", {}).get("load_avg")
    lf_scale = lf_js.get("context", {}).get("cpu_scaling_enabled")
    lb_scale = lb_js.get("context", {}).get("cpu_scaling_enabled")
    lines.append(
        "Normalized to same-run std::atomic<uint64_t>::compare_exchange_strong (contended)."
    )
    lines.append(
        "Do not compare raw ns across the two processes. Do not publish geomean."
    )
    lines.append(f"lock-free  load_avg={lf_load} cpu_scaling={lf_scale}")
    lines.append(f"lock-based load_avg={lb_load} cpu_scaling={lb_scale}")
    if lf_scale or lb_scale:
        lines.append(
            "CPU scaling is enabled; pin with: sudo cpupower frequency-set -g performance"
        )
    header = "threads | " + " | ".join(str(t) for t in threads)
    sep = "|".join(["---"] * (len(threads) + 1))

    lines.append("")
    lines.append("uint64 CAS contended (ns, denominator):")
    lines.append(header)
    lines.append(sep)
    lines.append("lock-free | " + " | ".join(f"{lf_base[t]:.2f}" for t in threads))
    lines.append("lock-based | " + " | ".join(f"{lb_base[t]:.2f}" for t in threads))
    lines.append(
        "lock-based/lock-free | "
        + " | ".join(f"{lb_base[t] / lf_base[t]:.2f}" for t in threads)
    )

    for op in SP_CONTENDED_OPS:
        prefix = f"std::atomic<shared_ptr<T>>::{op}() (contended)"
        lf_row = times_by_threads(lf, prefix)
        lb_row = times_by_threads(lb, prefix)
        lf_x = [(lf_row[t] / lf_base[t]) if t in lf_row else None for t in threads]
        lb_x = [(lb_row[t] / lb_base[t]) if t in lb_row else None for t in threads]
        lines.append("")
        lines.append(f"{op}() contended / uint64 CAS:")
        lines.append(header)
        lines.append(sep)
        lines.append("lock-free | " + format_ratio_row(lf_x))
        lines.append("lock-based | " + format_ratio_row(lb_x))

    lines.append("")
    lines.append(
        "Uncontended shared_ptr (raw ns; DWCAS bookkeeping is expected to lose):"
    )
    for name in SP_UNCONTENDED:
        lf_t = uncontended_time(lf, name)
        lb_t = uncontended_time(lb, name)
        short = name.removeprefix("std::atomic<shared_ptr<T>>::").removesuffix(
            " (uncontended)"
        )
        lf_s = f"{lf_t:.2f}" if lf_t is not None else "n/a"
        lb_s = f"{lb_t:.2f}" if lb_t is not None else "n/a"
        lines.append(f"  {short}: lock-free {lf_s}  lock-based {lb_s}")

    text = "\n".join(lines) + "\n"
    report_out.write_text(text, encoding="utf-8")
    sys.stdout.write(text)
    log(f"wrote {report_out}")


def write_lnt(benchmarks: list[dict[str, Any]], lnt_out: Path) -> None:
    rows = []
    for bm in benchmarks:
        name = strip_aggregate_suffix(bm["name"])
        escaped = name.replace(".", "_").replace(" ", "_")
        rows.append(f"{escaped}.execution_time {bm['real_time']}")
    lnt_out.write_text("\n".join(rows) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class ChartSeries:
    """One line in a chart: a label, a color, and threads -> value samples."""

    name: str
    color: str
    threads: list[int]
    values: list[float]


@dataclass(frozen=True)
class ChartSpec:
    """Everything needed to render one lock-free vs lock-based comparison chart."""

    title: str
    y_title: str
    filename: str
    series: tuple[ChartSeries, ChartSeries]
    note: str = ""


class BenchmarkChartRenderer:
    """Renders ChartSpec objects as matplotlib line charts saved to PNG.

    Each chart plots one metric (baseline ns, or a ratio to baseline) against
    thread count, with one line per implementation: major gridlines on the
    labeled ticks, lighter dotted minor gridlines between them, and a
    bordered legend.
    """

    MAJOR_GRID_COLOR = "#d9d9d9"
    MINOR_GRID_COLOR = "#eeeeee"
    DPI = 150
    FIGSIZE = (9.0, 5.5)

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def render(self, spec: ChartSpec) -> Path:
        import matplotlib

        matplotlib.use("Agg")  # headless: no display on a build/CI box
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker

        figure, axes = plt.subplots(figsize=self.FIGSIZE, dpi=self.DPI)
        all_threads: set[int] = set()
        for series in spec.series:
            all_threads.update(series.threads)
            axes.plot(
                series.threads,
                series.values,
                label=series.name,
                color=series.color,
                linewidth=2.5,
                marker="o",
                markersize=7,
                markeredgecolor="white",
                markeredgewidth=1,
            )

        axes.set_title(spec.title, fontsize=13, fontweight="bold")
        axes.set_xlabel("threads")
        axes.set_ylabel(spec.y_title)
        axes.xaxis.set_major_locator(mticker.FixedLocator(sorted(all_threads)))
        axes.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
        axes.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
        axes.set_axisbelow(True)
        axes.grid(True, which="major", color=self.MAJOR_GRID_COLOR, linewidth=0.9)
        axes.grid(
            True,
            which="minor",
            color=self.MINOR_GRID_COLOR,
            linewidth=0.6,
            linestyle=":",
        )
        for side in ("top", "right"):
            axes.spines[side].set_visible(False)
        legend = axes.legend(
            frameon=True, framealpha=0.9, edgecolor="#c9c9c9", loc="best"
        )
        legend.get_frame().set_linewidth(0.8)
        if spec.note:
            figure.text(0.01, 0.01, spec.note, fontsize=8.5, color="#666666", ha="left")
        figure.tight_layout(rect=(0, 0.04, 1, 1))

        out_path = self._output_dir / spec.filename
        figure.savefig(out_path)
        plt.close(figure)
        return out_path

    def render_all(self, specs: list[ChartSpec]) -> list[Path]:
        return [self.render(spec) for spec in specs]


def build_chart_specs(
    lf_benches: list[dict[str, Any]],
    lb_benches: list[dict[str, Any]],
    title_suffix: str,
    filename_suffix: str,
) -> list[ChartSpec]:
    """Same comparisons as print_normalized_report, as ChartSpecs instead of text."""
    lf_base = times_by_threads(lf_benches, U64_CAS_CONTENDED)
    lb_base = times_by_threads(lb_benches, U64_CAS_CONTENDED)
    threads = sorted(set(lf_base) & set(lb_base))
    if not threads:
        return []

    specs = [
        ChartSpec(
            title=f"uint64 CAS contended (ns, baseline){title_suffix}",
            y_title="ns/op",
            filename=f"baseline_cas{filename_suffix}.png",
            series=(
                ChartSeries(
                    "lock-free", LOCK_FREE_COLOR, threads, [lf_base[t] for t in threads]
                ),
                ChartSeries(
                    "lock-based",
                    LOCK_BASED_COLOR,
                    threads,
                    [lb_base[t] for t in threads],
                ),
            ),
            note="Denominator for the ratio charts below; same code in both binaries, should track near 1x.",
        )
    ]

    for op in SP_CONTENDED_OPS:
        prefix = f"std::atomic<shared_ptr<T>>::{op}() (contended)"
        lf_row = times_by_threads(lf_benches, prefix)
        lb_row = times_by_threads(lb_benches, prefix)
        op_threads = sorted(t for t in threads if t in lf_row and t in lb_row)
        if not op_threads:
            continue
        specs.append(
            ChartSpec(
                title=f"{op}() contended / uint64 CAS{title_suffix}",
                y_title="ratio to baseline CAS",
                filename=f"{op}_ratio{filename_suffix}.png",
                series=(
                    ChartSeries(
                        "lock-free",
                        LOCK_FREE_COLOR,
                        op_threads,
                        [lf_row[t] / lf_base[t] for t in op_threads],
                    ),
                    ChartSeries(
                        "lock-based",
                        LOCK_BASED_COLOR,
                        op_threads,
                        [lb_row[t] / lb_base[t] for t in op_threads],
                    ),
                ),
                note="Do not compare raw ns across the two processes; only this ratio.",
            )
        )
    return specs


def render_charts(
    lf_json: Path,
    lb_json: Path,
    charts_dir: Path,
    repetitions: int,
    separate_charts: bool,
) -> None:
    """Write the aggregate comparison charts, and per-repetition ones if asked.

    With --separate-charts and --repetitions N, this writes N per-repetition
    chart sets plus one aggregate (median) set per metric - N+1 charts per
    metric in total. Without --separate-charts, only the aggregate set.
    """
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        log(
            "charts skipped: matplotlib not installed "
            "(pip install -r libcxx/utils/requirements.txt, or pass --no-chart)"
        )
        return

    lf_js = json.loads(lf_json.read_text(encoding="utf-8"))
    lb_js = json.loads(lb_json.read_text(encoding="utf-8"))
    renderer = BenchmarkChartRenderer(charts_dir)

    agg_suffix = " (median)" if repetitions > 1 else ""
    agg_specs = build_chart_specs(
        selected_benchmarks(lf_js), selected_benchmarks(lb_js), agg_suffix, ""
    )
    for path in renderer.render_all(agg_specs):
        log(f"wrote chart {path}")

    if not separate_charts:
        return
    if repetitions <= 1:
        log(
            "--separate-charts has nothing to separate at --repetitions 1 (only the aggregate chart was written)"
        )
        return
    for rep in range(repetitions):
        lf_rep = benchmarks_for_repetition(lf_js, rep)
        lb_rep = benchmarks_for_repetition(lb_js, rep)
        rep_specs = build_chart_specs(
            lf_rep, lb_rep, f" (repetition {rep + 1}/{repetitions})", f"_rep{rep + 1}"
        )
        for path in renderer.render_all(rep_specs):
            log(f"wrote chart {path}")


def find_repo_root(start: Path) -> Path:
    for directory in [start, *start.parents]:
        if (directory / LIBCXX_LIT).is_file():
            return directory
    die(f"cannot find llvm-project root from {start}")


def libcxx_test_roots(repo: Path, build: Path) -> list[Path]:
    return [
        build / BOOTSTRAP_LIBCXX_TEST,
        build / LIBCXX_TEST,
        repo / LIBCXX_TEST,
    ]


def normalize_threads(raw: str) -> str:
    compact = re.sub(r"\s+", "", raw)
    if not compact:
        die("--threads is empty")
    if not THREADS_RE.fullmatch(compact):
        die(
            f"invalid --threads '{raw}' "
            "(expected e.g. 2,4,8 or 2,4,6,8,10,12,14,16,18,20)"
        )
    seen = []
    max_n = 0
    for part in compact.split(","):
        value = int(part)
        if value > MAX_THREADS:
            die(f"thread count {value} exceeds {MAX_THREADS}")
        if value not in seen:
            seen.append(value)
        max_n = max(max_n, value)
    cpus = os.cpu_count() or 0
    if cpus and max_n > cpus:
        log(
            f"warning: max thread count {max_n} > os.cpu_count()={cpus} (oversubscription)"
        )
    return ",".join(str(v) for v in seen)


def is_bootstrap_build(build: Path) -> bool:
    return (build / BOOTSTRAP_LIBCXX_TEST).is_dir()


def lit_test_path(repo: Path, build: Path, name: str) -> Path:
    root = (
        build / BOOTSTRAP_LIBCXX_TEST
        if is_bootstrap_build(build)
        else repo / LIBCXX_TEST
    )
    return root / BENCHMARKS_DIR / name


def find_output_dir(repo: Path, build: Path, name: str) -> Path | None:
    for root in libcxx_test_roots(repo, build):
        path = root / LIT_OUTPUT_DIR / f"{name}.dir"
        if path.is_dir():
            return path
    return None


def bench_inputs(repo: Path, src_name: str) -> list[Path]:
    return [
        repo / LIBCXX_TEST / BENCHMARKS_DIR / src_name,
        repo / LIBCXX_TEST / BENCH_HEADER,
        *[repo / rel for rel in ATOMIC_SP_HEADERS],
    ]


def exe_is_fresh(repo: Path, src_name: str, exe: Path) -> bool:
    if not exe.is_file():
        return False
    exe_mtime = exe.stat().st_mtime
    for src in bench_inputs(repo, src_name):
        if src.is_file() and src.stat().st_mtime > exe_mtime:
            return False
    return True


def find_bench_exe(output_dir: Path) -> Path | None:
    named = output_dir / LIT_BENCH_EXE
    if named.is_file() and os.access(named, os.X_OK):
        return named
    matches = sorted(output_dir.glob("*.exe"))
    if matches:
        return matches[0]
    return None


def elf_runpath_dirs(exe: Path) -> list[str]:
    readelf = shutil.which("readelf") or shutil.which("llvm-readelf")
    if readelf is None:
        return []
    try:
        out = subprocess.check_output(
            [readelf, "-d", str(exe)],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    dirs: list[str] = []
    origin = str(exe.parent)
    for line in out.splitlines():
        if "RPATH" not in line and "RUNPATH" not in line:
            continue
        start = line.find("[")
        end = line.rfind("]")
        if start == -1 or end <= start:
            continue
        for part in line[start + 1 : end].split(":"):  # noqa: E203
            part = part.replace("$ORIGIN", origin).replace("${ORIGIN}", origin)
            if part and part not in dirs:
                dirs.append(part)
    return dirs


def infer_libcxx_lib_dirs(exe: Path) -> list[str]:
    dirs: list[str] = []
    for parent in exe.parents:
        install_lib = parent / "test-suite-install" / "lib"
        if not install_lib.is_dir():
            continue
        if (install_lib / "libc++.so.1").is_file() or (
            install_lib / "libc++.so"
        ).is_file():
            dirs.append(str(install_lib))
        for sub in sorted(install_lib.iterdir()):
            if not sub.is_dir():
                continue
            if (sub / "libc++.so.1").is_file() or (sub / "libc++.so").is_file():
                dirs.append(str(sub))
    return dirs


def env_for_exe(base: dict[str, str], exe: Path) -> dict[str, str]:
    """Prepend the exe RUNPATH so LD_LIBRARY_PATH cannot pick system libc++."""
    env = base.copy()
    extra = elf_runpath_dirs(exe) + infer_libcxx_lib_dirs(exe)
    if not extra:
        return env
    key = "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"
    seen: list[str] = []
    for directory in extra:
        if directory not in seen:
            seen.append(directory)
    prefix = os.pathsep.join(seen)
    old = env.get(key, "")
    env[key] = prefix + (os.pathsep + old if old else "")
    log(f"{key} prefix={prefix}")
    return env


def run_command(
    argv: list[str],
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> None:
    log(" ".join(argv))
    try:
        subprocess.run(argv, check=True, env=env, cwd=cwd)
    except FileNotFoundError as exc:
        die(f"command not found: {argv[0]} ({exc})")
    except subprocess.CalledProcessError as exc:
        die(f"command failed with exit {exc.returncode}: {argv[0]}", exc.returncode)


def python_tool(repo: Path, name: str) -> list[str]:
    script = repo / LIBCXX_UTILS / name
    if not script.is_file():
        die(f"missing {script}")
    return [sys.executable, str(script)]


def build_lit_command(
    repo: Path,
    build: Path,
    test_path: Path,
    min_time: str,
    extra: list[str],
) -> list[str]:
    llvm_lit = build / LLVM_LIT
    if os.name != "nt":
        lit = repo / LIBCXX_LIT
        if not os.access(lit, os.X_OK):
            die(f"missing executable {lit}")
        cmd = [str(lit)]
        if is_bootstrap_build(build):
            cmd.append("-b")
        cmd.append(str(build))
    else:
        if not llvm_lit.is_file():
            die(f"llvm-lit not found: {llvm_lit}")
        cmake = shutil.which("cmake") or os.environ.get("CMAKE", "cmake")
        target = (
            "runtimes-test-depends" if is_bootstrap_build(build) else "cxx-test-depends"
        )
        run_command([cmake, "--build", str(build), "--target", target])
        cmd = [str(llvm_lit)]
    cmd += [
        "--show-all",
        "--param",
        "optimization=speed",
        "--param",
        "enable_benchmarks=dry-run",
        "--param",
        f"benchmark_min_time={min_time}",
        str(test_path),
        *extra,
    ]
    return cmd


def validate_bench_output(exe: Path, text: str, threads: str, label: str) -> None:
    """Catch the two ways a run can silently measure the wrong thing."""
    missing = [part for part in threads.split(",") if f"threads:{part}" not in text]
    if missing:
        die(
            f"{exe.name} did not register threads {','.join(missing)} "
            f"(asked {THREADS_ENV}={threads}). "
            "libcxx/utils/run.py uses a clean env; this script must run "
            "t.tmp.exe itself, not rely on lit %{exec}."
        )
    if label == "lock-free" and "libcxx_lock_free" not in text:
        die(
            f"{exe.name} is not the lock-free path (no libcxx_lock_free label). "
            "On x86-64 Clang needs -march=x86-64-v2 (see atomic_shared_ptr.bench.cpp). "
            "Recompile with --rebuild after that flag is present."
        )


def ensure_bench_exe(
    repo: Path,
    build: Path,
    src_name: str,
    label: str,
    min_time: str,
    extra: list[str],
    env: dict[str, str],
    force_rebuild: bool,
) -> Path:
    """Return a fresh t.tmp.exe for src_name, compiling via lit if needed."""
    output_dir = find_output_dir(repo, build, src_name)
    exe = find_bench_exe(output_dir) if output_dir is not None else None
    if exe is not None and not force_rebuild and exe_is_fresh(repo, src_name, exe):
        log(f"reusing {exe} ({label}; skip lit / cmake)")
        return exe
    test_path = lit_test_path(repo, build, src_name)
    log(f"compiling {label} via lit dry-run")
    run_command(build_lit_command(repo, build, test_path, min_time, extra), env=env)
    output_dir = find_output_dir(repo, build, src_name)
    if output_dir is None:
        die(f"no Output/{src_name}.dir after lit; check the run above")
    exe = find_bench_exe(output_dir)
    if exe is None:
        die(f"no {LIT_BENCH_EXE} under {output_dir}")
    return exe


def run_exe_to_lnt(
    exe: Path,
    min_time: str,
    repetitions: int,
    filt: str | None,
    json_out: Path,
    lnt_out: Path,
    env: dict[str, str],
    threads: str,
    label: str,
) -> None:
    argv = [
        str(exe),
        f"--benchmark_min_time={min_time}",
        f"--benchmark_out={json_out}",
        "--benchmark_out_format=json",
    ]
    if repetitions > 1:
        argv.append(f"--benchmark_repetitions={repetitions}")
        # Keep individual repetition rows in the JSON, not just the aggregates:
        # --separate-charts needs them. write_lnt/selected_benchmarks() below
        # still filter down to the median-only view for the .lnt file.
    if filt:
        argv.append(f"--benchmark_filter={filt}")
    run_command(argv, env=env_for_exe(env, exe), cwd=exe.parent)
    text = json_out.read_text(encoding="utf-8")
    validate_bench_output(exe, text, threads, label)
    js = json.loads(text)
    scaling = js.get("context", {}).get("cpu_scaling_enabled")
    if scaling:
        log(f"{label}: CPU scaling is enabled (google-benchmark context)")
    write_lnt(selected_benchmarks(js), lnt_out)


def run_one_bench(
    repo: Path,
    build: Path,
    src_name: str,
    label: str,
    lnt_out: Path,
    min_time: str,
    repetitions: int,
    extra: list[str],
    env: dict[str, str],
    filt: str | None,
    force_rebuild: bool,
) -> None:
    threads = env[THREADS_ENV]
    exe = ensure_bench_exe(
        repo, build, src_name, label, min_time, extra, env, force_rebuild
    )
    json_out = lnt_out.with_suffix(".json")
    log(
        f"running {label}: {THREADS_ENV}={threads} min_time={min_time} "
        f"repetitions={repetitions}"
    )
    if filt:
        log(f"filter {label}: --benchmark_filter={filt}")
    run_exe_to_lnt(
        exe, min_time, repetitions, filt, json_out, lnt_out, env, threads, label
    )


def run_bench_once(
    exe: Path,
    min_time: str,
    filt: str | None,
    env: dict[str, str],
    threads: str,
    label: str,
    json_out: Path,
) -> dict[str, Any]:
    """Run exe for exactly one repetition (no --benchmark_repetitions) and
    return the parsed JSON. Used by run_interleaved, which needs to
    interleave individual repetitions across the two binaries rather than
    letting Google Benchmark run all repetitions of one binary back to back.
    """
    argv = [
        str(exe),
        f"--benchmark_min_time={min_time}",
        f"--benchmark_out={json_out}",
        "--benchmark_out_format=json",
    ]
    if filt:
        argv.append(f"--benchmark_filter={filt}")
    run_command(argv, env=env_for_exe(env, exe), cwd=exe.parent)
    text = json_out.read_text(encoding="utf-8")
    validate_bench_output(exe, text, threads, label)
    return json.loads(text)


def _safe_stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _safe_cv(values: list[float]) -> float:
    mean = statistics.mean(values)
    return (_safe_stdev(values) / mean) if mean else 0.0


def merge_repetition_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine N single-repetition Google Benchmark JSON runs into the same
    shape --benchmark_repetitions=N would have produced: every individual
    iteration row (tagged with its repetition_index, like run_exe_to_lnt's
    output already is) plus computed mean/median/stddev/cv aggregate rows.
    This is what lets selected_benchmarks(), times_by_threads(),
    print_normalized_report(), and the chart builders treat an interleaved
    run exactly like a normal --benchmark_repetitions run.
    """
    all_rows: list[dict[str, Any]] = []
    by_key: dict[str, list[dict[str, Any]]] = {}
    for rep_index, run_js in enumerate(runs):
        for bm in run_js["benchmarks"]:
            row = dict(bm)
            row["run_type"] = "iteration"
            row["repetition_index"] = rep_index
            row["repetitions"] = len(runs)
            all_rows.append(row)
            by_key.setdefault(row["name"], []).append(row)

    for name, rows in by_key.items():
        real_times = [float(r["real_time"]) for r in rows]
        cpu_times = [float(r["cpu_time"]) for r in rows]
        label = rows[0].get("label", "")
        time_unit = rows[0].get("time_unit", "ns")
        iterations = rows[0].get("iterations", len(rows))
        aggregates = (
            ("mean", statistics.mean(real_times), statistics.mean(cpu_times)),
            ("median", statistics.median(real_times), statistics.median(cpu_times)),
            ("stddev", _safe_stdev(real_times), _safe_stdev(cpu_times)),
            ("cv", _safe_cv(real_times), _safe_cv(cpu_times)),
        )
        for agg_name, real_value, cpu_value in aggregates:
            all_rows.append(
                {
                    "name": f"{name}_{agg_name}",
                    "run_type": "aggregate",
                    "aggregate_name": agg_name,
                    "repetitions": len(runs),
                    "iterations": iterations,
                    "real_time": real_value,
                    "cpu_time": cpu_value,
                    "time_unit": time_unit,
                    "label": label,
                }
            )

    context = dict(runs[0].get("context", {})) if runs else {}
    return {"context": context, "benchmarks": all_rows}


def run_interleaved(
    repo: Path,
    build: Path,
    min_time: str,
    repetitions: int,
    extra: list[str],
    env: dict[str, str],
    filt: str | None,
    force_rebuild: bool,
    swap_order: bool,
    settle_seconds: float,
    lf_lnt: Path,
    lb_lnt: Path,
) -> None:
    """Alternate lock-free/lock-based one repetition at a time (rep 1 of
    each, then rep 2 of each, ...) so residual thermal or clock-frequency
    drift lands on both sides evenly instead of systematically favoring
    whichever binary runs first (see --swap-order, --interleave help).
    """
    order = [
        (LOCK_FREE_SRC, "lock-free", lf_lnt),
        (LOCK_BASED_SRC, "lock-based", lb_lnt),
    ]
    if swap_order:
        order.reverse()

    exes = {
        label: ensure_bench_exe(
            repo, build, src, label, min_time, extra, env, force_rebuild
        )
        for src, label, _ in order
    }
    threads = env[THREADS_ENV]
    runs: dict[str, list[dict[str, Any]]] = {label: [] for _, label, _ in order}

    with tempfile.TemporaryDirectory(prefix="atomic_sp_interleave_") as scratch:
        first = True
        for rep in range(repetitions):
            for _src, label, _lnt in order:
                if not first and settle_seconds > 0:
                    log(f"settling {settle_seconds}s before {label}")
                    time.sleep(settle_seconds)
                first = False
                log(f"interleaved {rep + 1}/{repetitions}: running {label}")
                json_out = Path(scratch) / f"{label}_{rep}.json"
                runs[label].append(
                    run_bench_once(
                        exes[label], min_time, filt, env, threads, label, json_out
                    )
                )

    for _src, label, lnt_out in order:
        merged = merge_repetition_runs(runs[label])
        scaling = merged.get("context", {}).get("cpu_scaling_enabled")
        if scaling:
            log(f"{label}: CPU scaling is enabled (google-benchmark context)")
        json_out = lnt_out.with_suffix(".json")
        json_out.write_text(json.dumps(merged), encoding="utf-8")
        write_lnt(selected_benchmarks(merged), lnt_out)
        log(f"wrote {json_out} ({label}, interleaved x{repetitions})")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Same-machine A/B of libc++ lock-free vs lock-based "
        "std::atomic<std::shared_ptr<T>> benches.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  %(prog)s --build <build>
  %(prog)s --build <build> --threads 2,4,6,8,10,12,14,16,18,20 --min-time 1s
  %(prog)s --build <build> --threads 2,4,6,8,10,12,14,16,18,20 --min-time 1s --repetitions 5
  %(prog)s --build <build> --threads 8 --filter 'compare_exchange_strong.*contended'
  %(prog)s --build <build> --repetitions 10 --separate-charts
  %(prog)s --build <build> --no-chart
  %(prog)s --build <build> --charts-out-dir /tmp/atomic_sp_charts
  %(prog)s --build <build> --repetitions 10 --interleave
  %(prog)s --build <build> --repetitions 10 --out-dir /tmp/ab_normal
  %(prog)s --build <build> --repetitions 10 --out-dir /tmp/ab_swapped --swap-order

notes:
  Run order is a systematic bias, not noise. By default lock-free runs first
  and lock-based second, so the second binary is always measured on an
  already-warmed chip and any residual thermal/frequency drift always lands
  on the same side. --swap-order reverses the order so you can measure that
  bias (run both directions into separate --out-dir paths and compare the
  normalized tables); --interleave alternates the two binaries one repetition
  at a time so the drift is spread evenly across both and the bias cancels.
  Prefer --interleave for numbers that will be published; use --swap-order
  when you want the bias quantified rather than removed.
  Charts (normalized-ratio line charts, lock-free vs lock-based) are written
  as PNG to <charts-out-dir or out-dir>/charts/*.png by default; pass
  --no-chart to skip them, --separate-charts to additionally get one chart
  set per repetition, or --charts-out-dir to redirect the charts/ folder
  somewhere other than --out-dir. Needs matplotlib
  (libcxx/utils/requirements.txt); missing matplotlib only skips charts, it
  does not fail the run.
  Run lock-free and lock-based one file at a time. Consolidating both
  Output dirs together mixes identical bench names.
  Contended rows at lit's default 0.2s often look like plateaus
  (1e6 / 5e6 / 2.5e6). This wrapper defaults to 1s.
  --threads 8 (alone) is the cheap isolation check: no threads:2/4 in
  the same process, so leaked static state cannot explain the number.
  After the first compile, omit --rebuild so only t.tmp.exe is re-run
  (libcxx-lit otherwise rebuilds runtimes-test-depends every time).
  Direct t.tmp.exe runs prepend the binary RUNPATH to LD_LIBRARY_PATH
  so a user path like /usr/lib cannot hide the just-built libc++.
  Publish from --repetitions 5 (median in LNT / normalized.txt). Single
  samples plus CPU scaling are not enough for an RFC table.
  compare-benchmarks prints a Geomean that mixes baseline rows with
  shared_ptr rows; do not publish it. Use the normalized table.
""",
    )
    parser.add_argument(
        "--build",
        type=Path,
        default=None,
        help="Build directory (default: $ATOMIC_SP_BENCH_BUILD or <repo>/build)",
    )
    parser.add_argument(
        "--threads",
        default=None,
        help=f"Comma-separated positive integers (default: {DEFAULT_THREADS})",
    )
    parser.add_argument(
        "--min-time",
        default=DEFAULT_MIN_TIME,
        help=f"lit --param benchmark_min_time (default: {DEFAULT_MIN_TIME})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "atomic_shared_ptr_bench",
        help="Where to write lock_free.lnt / lock_based.lnt",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=DEFAULT_REPETITIONS,
        help=(
            "Google Benchmark --benchmark_repetitions. Values >1 keep the median "
            "in LNT / normalized.txt / the aggregate charts, and also keep every "
            "individual repetition in the JSON for --separate-charts. "
            f"Default: {DEFAULT_REPETITIONS}. Use 5 before publishing."
        ),
    )
    parser.add_argument(
        "--filter",
        default=None,
        help="Pass --benchmark_filter to t.tmp.exe (after compile)",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force lit dry-run compile even if t.tmp.exe is newer than sources",
    )
    parser.add_argument("--lock-free-only", action="store_true")
    parser.add_argument("--lock-based-only", action="store_true")
    parser.add_argument("--no-compare", action="store_true")
    parser.add_argument(
        "--no-chart",
        action="store_true",
        help="Skip rendering charts (normalized.txt / LNT are unaffected)",
    )
    parser.add_argument(
        "--separate-charts",
        action="store_true",
        help=(
            "Also render one chart set per repetition, in addition to the "
            "aggregate (median) one - e.g. --repetitions 10 writes 10 "
            "per-repetition chart sets plus 1 aggregate set per metric. "
            "Needs --repetitions > 1; no-op otherwise."
        ),
    )
    parser.add_argument(
        "--charts-out-dir",
        type=Path,
        default=None,
        help="Parent directory for the charts/ folder (default: --out-dir)",
    )
    parser.add_argument(
        "--swap-order",
        action="store_true",
        help=(
            "Run lock-based first and lock-free second, reversing the default "
            "order. By default this script always runs lock-free first, so the "
            "second binary is always measured on an already-warmed chip; any "
            "residual thermal or clock drift that the CPU governor does not "
            "remove therefore always lands on the same side of the comparison, "
            "which is a systematic bias, not random noise. Use this to measure "
            "that bias: run once normally, once with --swap-order (writing to "
            "different --out-dir paths), and compare the two normalized tables. "
            "Matching ratios mean run order does not affect the result and the "
            "numbers are safe to publish; diverging ratios give you the "
            "magnitude of the order effect to quote as an error bar. Cheaper "
            "than --interleave (two normal runs), but only quantifies the bias "
            "instead of cancelling it."
        ),
    )
    parser.add_argument(
        "--interleave",
        action="store_true",
        help=(
            "Alternate the two binaries one repetition at a time (lock-free "
            "rep 1, lock-based rep 1, lock-free rep 2, lock-based rep 2, ...) "
            "instead of running all repetitions of one binary and then all "
            "repetitions of the other. This spreads thermal and frequency "
            "drift evenly across both sides rather than concentrating it in "
            "whichever binary runs second, so it cancels the order bias that "
            "--swap-order only measures. Needs --repetitions > 1 to do "
            "anything. Each repetition is a separate process launch, so the "
            "individual runs are recombined into the usual per-repetition plus "
            "mean/median/stddev/cv layout; the resulting .json / .lnt / "
            "normalized.txt / charts are identical in shape to a normal run, "
            "including --separate-charts. Slower to start (2*N process "
            "launches instead of 2) but the preferred mode for numbers that "
            "will be published. Combine with --swap-order to flip which "
            "binary leads each interleaved pair."
        ),
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=0.0,
        help=(
            "Sleep this many seconds every time execution switches from one "
            "binary to the other, to let the chip cool back toward a baseline "
            "thermal/frequency state before the next measurement. In "
            "sequential mode (the default, or with --swap-order) that is a "
            "single pause before the second binary; with --interleave it is "
            "before every switch, so total added wall-clock time is roughly "
            "2*(--repetitions)*--settle-seconds. Cheap and does not change the "
            "script's structure, but only reduces the thermal effect instead "
            "of measuring (--swap-order) or cancelling (--interleave) it - use "
            "it together with those, not instead of them. Default: 0 (no "
            "pause)."
        ),
    )
    parser.add_argument(
        "lit_args",
        nargs=argparse.REMAINDER,
        help="Extra llvm-lit args after --",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    start_time = time.monotonic()
    args = parse_args(argv)
    repo = find_repo_root(Path(__file__).resolve().parent)

    build = args.build
    if build is None:
        env_build = os.environ.get("ATOMIC_SP_BENCH_BUILD")
        build = Path(env_build) if env_build else repo / "build"
    build = build.resolve()
    if not build.is_dir():
        die(f"build directory not found: {build}")
    if not (build / LLVM_LIT).exists():
        die(
            f"llvm-lit not found under {build / LLVM_LIT.parent}; configure the build first"
        )

    extra = list(args.lit_args)
    if extra and extra[0] == "--":
        extra = extra[1:]

    threads = args.threads
    if threads is None:
        threads = os.environ.get(THREADS_ENV, DEFAULT_THREADS)
    threads = normalize_threads(threads)

    if not MIN_TIME_RE.fullmatch(args.min_time):
        die(f"invalid --min-time '{args.min_time}' (expected e.g. 1s, 0.5s, 500ms)")
    if args.repetitions < 1:
        die(f"invalid --repetitions {args.repetitions} (expected >= 1)")
    if args.settle_seconds < 0:
        die(f"invalid --settle-seconds {args.settle_seconds} (expected >= 0)")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir = out_dir.resolve()

    charts_parent = args.charts_out_dir if args.charts_out_dir is not None else out_dir
    charts_parent.mkdir(parents=True, exist_ok=True)
    charts_parent = charts_parent.resolve()

    env = os.environ.copy()
    env[THREADS_ENV] = threads

    do_lock_free = not args.lock_based_only
    do_lock_based = not args.lock_free_only
    if args.lock_free_only and args.lock_based_only:
        die("use only one of --lock-free-only / --lock-based-only")

    interleave = args.interleave and do_lock_free and do_lock_based
    if args.interleave and not interleave:
        log("--interleave needs both implementations; ignoring it with --lock-*-only")
    if interleave and args.repetitions <= 1:
        log("--interleave with --repetitions 1 is just a plain sequential run")

    log(f"repo={repo}")
    log(f"build={build} bootstrap={'yes' if is_bootstrap_build(build) else 'no'}")
    log(
        f"threads={threads} min_time={args.min_time} "
        f"repetitions={args.repetitions} out={out_dir} "
        f"charts={'(disabled)' if args.no_chart else charts_parent / CHARTS_DIR_NAME}"
    )
    log(
        f"order={'lock-based first' if args.swap_order else 'lock-free first'} "
        f"mode={'interleaved' if interleave else 'sequential'}"
    )

    lf_lnt = out_dir / "lock_free.lnt"
    lb_lnt = out_dir / "lock_based.lnt"

    if interleave:
        run_interleaved(
            repo,
            build,
            args.min_time,
            args.repetitions,
            extra,
            env,
            args.filter,
            args.rebuild,
            args.swap_order,
            args.settle_seconds,
            lf_lnt,
            lb_lnt,
        )
    else:
        planned = [
            (do_lock_free, LOCK_FREE_SRC, "lock-free", lf_lnt),
            (do_lock_based, LOCK_BASED_SRC, "lock-based", lb_lnt),
        ]
        if args.swap_order:
            planned.reverse()
        first = True
        for enabled, src, label, lnt_out in planned:
            if not enabled:
                continue
            if not first and args.settle_seconds > 0:
                log(f"settling {args.settle_seconds}s before {label}")
                time.sleep(args.settle_seconds)
            first = False
            run_one_bench(
                repo,
                build,
                src,
                label,
                lnt_out,
                args.min_time,
                args.repetitions,
                extra,
                env,
                args.filter,
                args.rebuild,
            )

    lf_json = lf_lnt.with_suffix(".json")
    lb_json = lb_lnt.with_suffix(".json")
    if do_lock_free and do_lock_based and lf_json.is_file() and lb_json.is_file():
        print_normalized_report(lf_json, lb_json, out_dir / "normalized.txt")
        if not args.no_chart:
            render_charts(
                lf_json,
                lb_json,
                charts_parent / CHARTS_DIR_NAME,
                args.repetitions,
                args.separate_charts,
            )

    if not args.no_compare and do_lock_free and do_lock_based:
        compare = python_tool(repo, "compare-benchmarks")
        log(f"compare-benchmarks {lf_lnt} {lb_lnt}")
        log(
            "compare-benchmarks Geomean mixes baseline and shared_ptr rows; "
            "do not publish it. Use normalized.txt."
        )
        run_command([*compare, str(lf_lnt), str(lb_lnt)])

    elapsed = time.monotonic() - start_time
    log(f"done in {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except KeyboardInterrupt:
        log("Interrupted")
        sys.exit(1)

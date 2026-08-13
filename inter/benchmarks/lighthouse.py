#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import shutil
import statistics
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
INTER_ROOT = HERE.parent
INTER_SOURCE = (
    INTER_ROOT / "test" / "Integration" / "Inputs" / "lighthouse-matmul.mlir"
)
IGC_SOURCE = HERE / "lighthouse-matmul.cl"
RESULT = re.compile(r"^(inter|igc) median_us=([0-9.]+).*$")


def run(command: list[str], *, capture: bool = False) -> str:
    process = subprocess.run(command, text=True, capture_output=capture)
    if process.returncode != 0:
        if capture:
            sys.stdout.write(process.stdout)
            sys.stderr.write(process.stderr)
        raise SystemExit(process.returncode)
    return process.stdout


def require(path: Path) -> Path:
    if not path.exists():
        raise SystemExit(f"required file is missing: {path}")
    return path


def build_binaries(
    build_dir: Path, output_dir: Path, igc_device: str, size: int
) -> tuple[Path, Path, str]:
    inter_opt = require(build_dir / "tools" / "inter-opt" / "inter-opt")
    inter_translate = require(
        build_dir / "tools" / "inter-translate" / "inter-translate"
    )
    pipelines = require(build_dir / "share" / "inter" / "pipelines" / "pipelines.mlir")
    output_dir.mkdir(parents=True, exist_ok=True)
    machine = output_dir / "lighthouse-inter.mlir"
    inter_binary = output_dir / "lighthouse-inter.bin"
    inter_source = output_dir / "lighthouse-input.mlir"
    source_text = INTER_SOURCE.read_text()
    replacements = {
        "%18 = llvm.mlir.constant(512 : i32) : i32": (
            f"%18 = llvm.mlir.constant({size * 4} : i32) : i32\n"
            f"    %matrix_size = llvm.mlir.constant({size} : i32) : i32"
        ),
        "%15 = llvm.mlir.constant(256 : i32) : i32": (
            f"%15 = llvm.mlir.constant({size * 2} : i32) : i32"
        ),
        "(%45, %7, %7, %7, %47)": "(%45, %7, %matrix_size, %7, %47)",
        "(%45, %7, %7, %7, %88)": "(%45, %7, %matrix_size, %7, %88)",
        "(%45, %7, %7, %7, %93, %94)": (
            "(%45, %7, %matrix_size, %7, %93, %94)"
        ),
        "(%45, %7, %7, %7, %100, %101)": (
            "(%45, %7, %matrix_size, %7, %100, %101)"
        ),
        "(%45, %7, %7, %7, %106, %107)": (
            "(%45, %7, %matrix_size, %7, %106, %107)"
        ),
        "(%45, %7, %7, %7, %111, %112)": (
            "(%45, %7, %matrix_size, %7, %111, %112)"
        ),
        "(%149, %18, %7, %18, %152, %153)": (
            "(%149, %18, %matrix_size, %18, %152, %153)"
        ),
        "(%149, %18, %7, %18, %157, %158)": (
            "(%149, %18, %matrix_size, %18, %157, %158)"
        ),
    }
    for frozen, replacement in replacements.items():
        if source_text.count(frozen) != 1:
            raise SystemExit(f"frozen Inter input no longer has one {frozen!r}")
        source_text = source_text.replace(frozen, replacement)
    inter_source.write_text(source_text)
    pipeline = (
        "builtin.module("
        f"transform-preload-library{{transform-library-paths={pipelines}}},"
        "transform-interpreter{entry-point=inter_backend})"
    )
    run(
        [
            str(inter_opt),
            str(inter_source),
            f"--pass-pipeline={pipeline}",
            "-o",
            str(machine),
        ]
    )
    run(
        [
            str(inter_translate),
            str(machine),
            "--xemachine-to-zebin",
            "-o",
            str(inter_binary),
        ]
    )

    ocloc = shutil.which("ocloc")
    if not ocloc:
        raise SystemExit("required tool is missing: ocloc")
    ocloc_version = run([ocloc, "--version"], capture=True).strip()
    igc_dir = output_dir / "igc"
    if igc_dir.exists():
        shutil.rmtree(igc_dir)
    igc_dir.mkdir()
    run(
        [
            ocloc,
            "compile",
            "-file",
            str(IGC_SOURCE),
            "-device",
            igc_device,
            "-options",
            f"-DMATRIX_SIZE={size}",
            "-out_dir",
            str(igc_dir),
        ]
    )
    igc_binaries = list(igc_dir.glob("lighthouse-matmul_*.bin"))
    if len(igc_binaries) != 1:
        raise SystemExit(
            f"expected one IGC native binary in {igc_dir}, found {len(igc_binaries)}"
        )
    igc_binary = igc_binaries[0]
    return inter_binary, igc_binary, ocloc_version


def measure(
    runner: Path,
    compiler: str,
    binary: Path,
    device: str,
    warmups: int,
    batches: int,
    iterations: int,
    size: int,
) -> float:
    output = run(
        [
            str(runner),
            compiler,
            str(binary),
            device,
            str(warmups),
            str(batches),
            str(iterations),
            str(size),
            "payload_kernel",
        ],
        capture=True,
    )
    print(output, end="")
    match = RESULT.search(output)
    if not match:
        raise SystemExit(f"could not parse benchmark output: {output!r}")
    return float(match.group(2))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare Inter and IGC Lighthouse matmul performance on BMG"
    )
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="B60")
    parser.add_argument("--igc-device", default="bmg-g21")
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=200)
    parser.add_argument("--batches", type=int, default=15)
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args()
    if min(args.runs, args.warmups, args.batches, args.iterations) < 1:
        parser.error("run counts must be positive")
    if args.size < 64 or args.size % 64:
        parser.error("--size must be a positive multiple of 64")

    build_dir = args.build_dir.resolve()
    output_dir = (args.output_dir or build_dir / "benchmarks" / "lighthouse").resolve()
    runner = require(build_dir / "benchmarks" / "inter-lighthouse-benchmark")
    inter_binary, igc_binary, ocloc_version = build_binaries(
        build_dir, output_dir, args.igc_device, args.size
    )
    print(
        f"configuration device={args.device} igc_device={args.igc_device} "
        f"shape={args.size}x{args.size}x64 "
        f"runs={args.runs} warmups={args.warmups} batches={args.batches} "
        f"iterations={args.iterations} timestamp=level-zero-kernel "
        f"ocloc={ocloc_version}"
    )
    samples: dict[str, list[float]] = {"inter": [], "igc": []}
    for run_index in range(args.runs):
        order = ("inter", "igc") if run_index % 2 == 0 else ("igc", "inter")
        for compiler in order:
            binary = inter_binary if compiler == "inter" else igc_binary
            samples[compiler].append(
                measure(
                    runner,
                    compiler,
                    binary,
                    args.device,
                    args.warmups,
                    args.batches,
                    args.iterations,
                    args.size,
                )
            )

    inter_median = statistics.median(samples["inter"])
    igc_median = statistics.median(samples["igc"])
    print(
        f"summary inter_median_us={inter_median:.6f} "
        f"inter_range_us={min(samples['inter']):.6f}..{max(samples['inter']):.6f} "
        f"igc_median_us={igc_median:.6f} "
        f"igc_range_us={min(samples['igc']):.6f}..{max(samples['igc']):.6f} "
        f"igc_speedup={inter_median / igc_median:.4f}x "
        f"inter_vs_igc={(inter_median / igc_median - 1.0) * 100.0:.2f}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

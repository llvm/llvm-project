#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

from lighthouse_input import specialize_inter_source

HERE = Path(__file__).resolve().parent
INTER_ROOT = HERE.parent
INTER_SOURCE = INTER_ROOT / "test" / "Integration" / "Inputs" / "lighthouse-matmul.mlir"
LIGHTHOUSE_REVISION = "ec3a77574cc5f049736f47b121bdd4aeeb854201"
RESULT = re.compile(r"^(inter|lighthouse) median_us=([0-9.]+).*$")
GPU_OBJECT = re.compile(
    r'gpu\.binary\s+@payload_kernel\b[^\n]*?#gpu\.object<[^\"]*?"'
    r'((?:\\[0-9A-Fa-f]{2}|\\.|[^"\\])*)"'
)


def run(
    command: list[str], *, capture: bool = False, timeout: float | None = None
) -> str:
    try:
        process = subprocess.run(
            command, text=True, capture_output=capture, timeout=timeout
        )
    except subprocess.TimeoutExpired as error:
        raise SystemExit(
            f"command timed out after {timeout:g}s: {' '.join(command)}"
        ) from error
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


def decode_mlir_string(value: str) -> bytes:
    result = bytearray()
    index = 0
    escapes = {"n": b"\n", "t": b"\t", '"': b'"', "\\": b"\\"}
    while index < len(value):
        if value[index] != "\\":
            result.extend(value[index].encode())
            index += 1
            continue
        if index + 2 < len(value) and all(
            character in "0123456789abcdefABCDEF"
            for character in value[index + 1 : index + 3]
        ):
            result.append(int(value[index + 1 : index + 3], 16))
            index += 3
            continue
        if index + 1 >= len(value) or value[index + 1] not in escapes:
            raise ValueError(f"unsupported MLIR string escape at byte {index}")
        result.extend(escapes[value[index + 1]])
        index += 2
    return bytes(result)


def build_lighthouse_binary(
    lighthouse_root: Path,
    lighthouse_python: Path,
    output_dir: Path,
    size: int,
    reduction_size: int,
) -> tuple[Path, str]:
    generator = require(lighthouse_root / "examples" / "xegpu" / "matmul.py")
    revision = run(
        ["git", "-C", str(lighthouse_root), "rev-parse", "HEAD"], capture=True
    ).strip()
    if revision != LIGHTHOUSE_REVISION:
        raise SystemExit(
            f"Lighthouse revision {revision} does not match the benchmark input's "
            f"pinned revision {LIGHTHOUSE_REVISION}"
        )
    command = [
        str(lighthouse_python),
        str(generator),
        "--sizes",
        str(size),
        str(size),
        str(reduction_size),
        "--wg-tile",
        "64",
        "64",
        "--sg-tile",
        "16",
        "16",
        "--k-tile",
        "32",
        "--load-tile-a",
        "8",
        "16",
        "--load-tile-b",
        "16",
        "16",
        "--prefetch-tile-a",
        "8",
        "16",
        "--prefetch-tile-b",
        "8",
        "16",
        "--prefetch-a-nb",
        "1",
        "--prefetch-b-nb",
        "1",
        "--dump-kernel=final",
        "--no-accumulate-c",
    ]
    try:
        process = subprocess.run(
            command,
            cwd=lighthouse_root,
            env=os.environ.copy(),
            text=True,
            stdout=subprocess.PIPE,
        )
    except OSError as error:
        raise SystemExit(f"failed to execute Lighthouse generator: {error}") from error
    if process.returncode != 0:
        raise SystemExit(process.returncode)
    generated_mlir = output_dir / "lighthouse-generated.mlir"
    generated_mlir.write_text(process.stdout)
    matches = GPU_OBJECT.findall(process.stdout)
    if len(matches) != 1:
        raise SystemExit(
            f"expected one payload_kernel GPU object from Lighthouse, found {len(matches)}"
        )
    try:
        binary = decode_mlir_string(matches[0])
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if not binary.startswith(b"\x7fELF"):
        raise SystemExit("Lighthouse GPU object is not an ELF binary")
    lighthouse_binary = output_dir / "lighthouse-native.bin"
    lighthouse_binary.write_bytes(binary)
    return lighthouse_binary, revision


def build_binaries(
    build_dir: Path,
    output_dir: Path,
    lighthouse_root: Path,
    lighthouse_python: Path,
    size: int,
    reduction_size: int,
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
    try:
        source_text = specialize_inter_source(
            INTER_SOURCE.read_text(), size, reduction_size
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
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

    lighthouse_binary, lighthouse_revision = build_lighthouse_binary(
        lighthouse_root,
        lighthouse_python,
        output_dir,
        size,
        reduction_size,
    )
    return inter_binary, lighthouse_binary, lighthouse_revision


def measure(
    runner: Path,
    compiler: str,
    binary: Path,
    device: str,
    warmups: int,
    batches: int,
    iterations: int,
    size: int,
    reduction_size: int,
    timeout: float,
    padding_k_tiles: int,
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
            str(reduction_size),
            "payload_kernel",
            str(padding_k_tiles),
        ],
        capture=True,
        timeout=timeout,
    )
    print(output, end="")
    match = RESULT.search(output)
    if not match:
        raise SystemExit(f"could not parse benchmark output: {output!r}")
    return float(match.group(2))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare Inter and Lighthouse matmul performance on BMG"
    )
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="B60")
    parser.add_argument(
        "--lighthouse-root", type=Path, default=Path.home() / "llvm" / "lighthouse"
    )
    parser.add_argument("--lighthouse-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--reduction-size", type=int, default=64)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=200)
    parser.add_argument("--batches", type=int, default=15)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--padding-k-tiles", type=int, default=0)
    args = parser.parse_args()
    if min(args.runs, args.warmups, args.batches, args.iterations) < 1:
        parser.error("run counts must be positive")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.padding_k_tiles < 0:
        parser.error("--padding-k-tiles must be nonnegative")
    if args.size < 64 or args.size % 64:
        parser.error("--size must be a positive multiple of 64")
    if args.reduction_size < 32 or args.reduction_size % 32:
        parser.error("--reduction-size must be a positive multiple of 32")

    build_dir = args.build_dir.resolve()
    output_dir = (args.output_dir or build_dir / "benchmarks" / "lighthouse").resolve()
    runner = require(build_dir / "benchmarks" / "inter-lighthouse-benchmark")
    inter_binary, lighthouse_binary, lighthouse_revision = build_binaries(
        build_dir,
        output_dir,
        args.lighthouse_root.resolve(),
        args.lighthouse_python.resolve(),
        args.size,
        args.reduction_size,
    )
    print(
        f"configuration device={args.device} "
        f"shape={args.size}x{args.size}x{args.reduction_size} "
        f"runs={args.runs} warmups={args.warmups} batches={args.batches} "
        f"iterations={args.iterations} timestamp=level-zero-kernel "
        f"timeout={args.timeout:g}s "
        f"padding_k_tiles={args.padding_k_tiles} "
        f"lighthouse_revision={lighthouse_revision}"
    )
    samples: dict[str, list[float]] = {"inter": [], "lighthouse": []}
    for run_index in range(args.runs):
        order = (
            ("inter", "lighthouse") if run_index % 2 == 0 else ("lighthouse", "inter")
        )
        for compiler in order:
            binary = inter_binary if compiler == "inter" else lighthouse_binary
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
                    args.reduction_size,
                    args.timeout,
                    args.padding_k_tiles,
                )
            )

    inter_median = statistics.median(samples["inter"])
    lighthouse_median = statistics.median(samples["lighthouse"])
    print(
        f"summary inter_median_us={inter_median:.6f} "
        f"inter_range_us={min(samples['inter']):.6f}..{max(samples['inter']):.6f} "
        f"lighthouse_median_us={lighthouse_median:.6f} "
        f"lighthouse_range_us={min(samples['lighthouse']):.6f}.."
        f"{max(samples['lighthouse']):.6f} "
        f"lighthouse_speedup={inter_median / lighthouse_median:.4f}x "
        f"inter_vs_lighthouse="
        f"{(inter_median / lighthouse_median - 1.0) * 100.0:.2f}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

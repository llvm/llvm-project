# RUN: %python %s --build-dir %inter_obj_root --pipelines %inter_pipelines --generated-out %t.s | FileCheck %s

# CHECK: asm-golden: lighthouse-matmul: asm matches golden

from __future__ import annotations

import argparse
import difflib
import subprocess
import sys
import tempfile
from pathlib import Path


NAME = "lighthouse-matmul"
HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "Integration" / "Inputs" / "lighthouse-matmul.mlir"
GOLDEN = HERE / "Inputs" / f"{NAME}.s"


def normalize_asm(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines) + "\n"


def generate_asm(build_dir: Path, pipelines: Path, output: Path) -> str:
    inter_opt = build_dir / "tools" / "inter-opt" / "inter-opt"
    inter_translate = build_dir / "tools" / "inter-translate" / "inter-translate"
    for tool in (inter_opt, inter_translate):
        if not tool.exists():
            raise SystemExit(f"required tool missing: {tool}")

    pipeline = (
        "builtin.module("
        f"transform-preload-library{{transform-library-paths={pipelines}}},"
        "transform-interpreter{entry-point=inter_backend})"
    )
    lowered = subprocess.run(
        [str(inter_opt), str(SOURCE), f"--pass-pipeline={pipeline}"],
        capture_output=True,
        check=False,
    )
    if lowered.returncode != 0:
        sys.stderr.buffer.write(lowered.stderr)
        raise SystemExit(lowered.returncode)
    emitted = subprocess.run(
        [str(inter_translate), "--xemachine-to-asm", "-"],
        input=lowered.stdout,
        capture_output=True,
        check=False,
    )
    if emitted.returncode != 0:
        sys.stderr.buffer.write(emitted.stderr)
        raise SystemExit(emitted.returncode)
    assembly = normalize_asm(emitted.stdout.decode("utf-8"))
    output.write_text(assembly, encoding="utf-8")
    return assembly


def print_diff(golden: str, generated: str, output: Path, max_lines: int) -> None:
    diff = list(
        difflib.unified_diff(
            golden.splitlines(),
            generated.splitlines(),
            fromfile=str(GOLDEN),
            tofile=str(output),
            lineterm="",
            n=3,
        )
    )
    if max_lines >= 0 and len(diff) > max_lines:
        diff = [*diff[:max_lines], f"... {len(diff) - max_lines} diff lines omitted"]
    for line in diff:
        print(line)


def check_asm(
    build_dir: Path,
    pipelines: Path,
    generated_out: Path | None,
    max_diff_lines: int,
) -> None:
    with tempfile.TemporaryDirectory() as directory:
        output = generated_out or Path(directory) / f"{NAME}.s"
        generated = normalize_asm(generate_asm(build_dir, pipelines, output))
        golden = normalize_asm(GOLDEN.read_text(encoding="utf-8"))
        if generated == golden:
            print(f"asm-golden: {NAME}: asm matches golden")
            return

        print(f"asm-golden: {NAME}: ASM DRIFT DETECTED")
        print(f"golden: {GOLDEN}")
        print(f"generated: {output}")
        print_diff(golden, generated, output, max_diff_lines)
        raise SystemExit(1)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--pipelines", type=Path, required=True)
    parser.add_argument("--generated-out", type=Path)
    parser.add_argument("--max-diff-lines", type=int, default=200)
    args = parser.parse_args(argv)
    check_asm(
        args.build_dir, args.pipelines, args.generated_out, args.max_diff_lines
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

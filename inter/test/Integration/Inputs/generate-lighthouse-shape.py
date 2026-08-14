from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarks"))

from lighthouse_input import specialize_inter_source


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("size", type=int)
    parser.add_argument("reduction_size", type=int)
    parser.add_argument("--drop-loop-prefetch", action="store_true")
    args = parser.parse_args()
    source_text = specialize_inter_source(
        args.source.read_text(), args.size, args.reduction_size
    )
    if args.drop_loop_prefetch:
        in_loop = False
        removed = 0
        lines = []
        for line in source_text.splitlines():
            in_loop |= line.startswith("  ^bb1")
            if (
                in_loop
                and line.lstrip().startswith("llvm.call")
                and "intel_sub_group_2d_block_prefetch" in line
            ):
                removed += 1
                continue
            lines.append(line)
        if removed != 2:
            raise ValueError(f"expected two loop prefetches, found {removed}")
        source_text = "\n".join(lines) + "\n"
    args.output.write_text(source_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

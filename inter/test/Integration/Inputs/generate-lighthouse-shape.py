from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarks"))

from lighthouse_input import drop_loop_prefetches, specialize_inter_source


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
        source_text = drop_loop_prefetches(source_text)
    args.output.write_text(source_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

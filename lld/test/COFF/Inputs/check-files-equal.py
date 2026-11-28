import pathlib
import sys

left = pathlib.Path(sys.argv[1]).read_bytes()
right = pathlib.Path(sys.argv[2]).read_bytes()
if left != right:
    raise SystemExit(f"files differ: {sys.argv[1]} and {sys.argv[2]}")

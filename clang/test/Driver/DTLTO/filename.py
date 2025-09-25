from pathlib import Path
import sys

print(f"filename.py:{Path(sys.argv[1]).resolve().name}")

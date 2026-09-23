#!/usr/bin/env python3
"""Refresh Clang's IPC2978 copy from the sibling ipc2978api repository."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    repo = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=repo.parent / 'ipc2978api')
    args = parser.parse_args()
    subprocess.run([
        sys.executable, str(args.source / 'tools/copy_library.py'),
        '--include-dir', str(repo / 'clang/include/clang/IPC2978'),
        '--source-dir', str(repo / 'clang/lib/IPC2978'),
        '--include-prefix', 'clang/IPC2978/',
        '--clang-tests-dir', str(repo / 'clang/unittests/IPC2978'),
    ], check=True)


if __name__ == '__main__':
    main()

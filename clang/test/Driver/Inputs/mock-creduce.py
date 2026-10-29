#!/usr/bin/env python3

import sys
import subprocess


def main():
    if len(sys.argv) < 3:
        sys.exit("Usage: mock-creduce.py [flags] <test_script> <file_to_reduce>")

    test_script = sys.argv[-2]
    file_to_reduce = sys.argv[-1]

    print(f"Mock creduce reducing {file_to_reduce} using {test_script}")

    with open(file_to_reduce, "r") as f:
        lines = f.readlines()

    # Simple line-by-line reduction
    i = 0
    while i < len(lines):
        # Try to remove line i
        candidate_lines = lines[:i] + lines[i + 1 :]

        # Run interestingness test
        with open(file_to_reduce, "w") as f:
            f.writelines(candidate_lines)
        res = subprocess.call([test_script])
        if res == 0:
            print(f"Mock creduce: Removed line {i}: {lines[i].strip()}")
            lines = candidate_lines
        else:
            with open(file_to_reduce, "w") as f:
                f.writelines(lines)
            i += 1

    print("Mock creduce finished.")


if __name__ == "__main__":
    main()

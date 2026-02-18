#!/usr/bin/env python3
"""Symbolize raw memprof stack addresses using llvm-symbolizer.

Usage: symbolize_raw_stacks.py <binary> <memprof_output_file>

Parses the module map from memprof output (requires print_module_map=1) to
determine the binary's ASLR slide, converts raw runtime addresses to
file-relative addresses, and symbolizes them using llvm-symbolizer (expected
to be on PATH). Prints one symbolized function name per line for addresses
that fall within the main binary.
"""

import re
import subprocess
import sys


def main():
    binary = sys.argv[1]
    output_file = sys.argv[2]

    with open(output_file) as f:
        data = f.read()

    # Parse module map to find the binary's load range.
    # Format: 0xSTART-0xEND /path/to/binary (arch) <UUID>
    binary_name = binary.rsplit("/", 1)[-1]
    slide = None
    end_addr = None
    for line in data.split("\n"):
        if binary_name in line and line.startswith("0x"):
            addr_range = line.split()[0]
            start_str, end_str = addr_range.split("-")
            slide = int(start_str, 16)
            end_addr = int(end_str, 16)
            break

    if slide is None:
        print("ERROR: Could not find binary in module map", file=sys.stderr)
        sys.exit(1)

    # Collect unique addresses within the binary's memory range.
    addrs = set()
    for match in re.finditer(r"#\d+ (0x[0-9a-f]+)", data):
        runtime_addr = int(match.group(1), 16)
        if slide <= runtime_addr <= end_addr:
            addrs.add(runtime_addr)

    # Symbolize using llvm-symbolizer with file-relative addresses.
    for runtime_addr in sorted(addrs):
        file_addr = runtime_addr - slide
        result = subprocess.run(
            ["llvm-symbolizer", "--obj=" + binary, hex(file_addr)],
            capture_output=True,
            text=True,
        )
        func_name = result.stdout.strip().split("\n")[0]
        if func_name and func_name != "??":
            print(func_name)


if __name__ == "__main__":
    main()

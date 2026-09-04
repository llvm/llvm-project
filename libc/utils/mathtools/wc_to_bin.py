#!/usr/bin/env python3
# ===-- Tool to convert CORE-MATH .wc files to raw binary data -------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===#

"""Convert CORE-MATH .wc (worst case) test files to raw IEEE 754 binary format.

This script parses text .wc files containing hexadecimal floating-point numbers,
handles comments, special values (NaNs, sNaNs, inf, signed zeros), and
outputs raw binary IEEE-754 values (by default 64-bit double precision,
little-endian).

Why binary?
  - CORE-MATH .wc files for binary64 functions often contain >1,000,000 cases,
    taking 25+ MB of text.
  - Raw binary representation reduces the file size by ~3x (e.g., 25 MB -> 8.7 MB).
  - Test suites can memory-map or read the entire binary file with a single fread()
    call in sub-milliseconds, avoiding expensive text parsing during test startup.

Supported Input Formats:
  - Hexadecimal floating point: '0x1.005023d32fee5p+1'
  - Standard decimal floats:    '1.2345'
  - Special values / NaNs:       'snan', '-snan', 'qnan', 'nan', 'inf', '-inf'
  - Signed zeros & integers:     '+0', '-0', '1', '-1'
  - Univariate inputs:           One value per line.
  - Bivariate inputs:            Values separated by commas, e.g. '0x1.0p+0, 0x1.8p+0'
  - Comments:                    Lines starting with '#' or inline '#' comments.

Example Usages:
  1. Convert a binary64 worst-case file to binary:
       $ python3 wc_to_bin.py sin.wc sin.bin

  2. Convert a binary32 (single-precision float) file:
       $ python3 wc_to_bin.py -t binary32 sinf.wc sinf.bin

  3. Use with stdin and stdout pipes:
       $ cat cos.wc | python3 wc_to_bin.py > cos.bin

  4. Convert and display summary statistics:
       $ python3 wc_to_bin.py --summary cos.wc cos.bin

TODO: Add a CI check to ensure that the .wc and .bin files in
libc/test/src/math/exhaustive/ are kept in sync.
"""

import argparse
import struct
import sys


def parse_token_binary64(s: str) -> int:
    """Parse a string token representing a binary64 number to a 64-bit unsigned integer.

    Returns the exact 64-bit IEEE 754 bit-pattern as an integer, or None if the token
    is empty.

    Handles:
      - Signaling NaNs ('snan', '+snan', '-snan')
      - Quiet NaNs ('qnan', 'nan', '+nan', '-nan', '-qnan')
      - Infinities ('inf', '+inf', '-inf', 'infinity', '-infinity')
      - Signed zeros ('+0', '-0', '0.0', '-0.0')
      - Common integer values ('1', '+1', '-1')
      - Hexadecimal floats ('0x1.0p+0') and standard floats
    """
    s = s.strip()
    if not s:
        return None

    # Handle signaling NaNs (quiet bit is 0, payload non-zero)
    if s in ("snan", "+snan"):
        return 0x7FF4000000000000
    if s == "-snan":
        return 0xFFF4000000000000

    # Handle quiet NaNs (quiet bit is 1)
    if s in ("qnan", "+qnan", "nan", "+nan"):
        return 0x7FF8000000000000
    if s in ("-nan", "-qnan"):
        return 0xFFF8000000000000

    # Handle infinities (exponent all 1s, mantissa 0)
    if s in ("inf", "+inf", "infinity", "+infinity"):
        return 0x7FF0000000000000
    if s in ("-inf", "-infinity"):
        return 0xFFF0000000000000

    # Handle zeros (preserve negative sign bit: 0x8000000000000000)
    if s in ("+0", "0", "+0.0", "0.0"):
        return 0x0000000000000000
    if s in ("-0", "-0.0"):
        return 0x8000000000000000

    # Handle common integer constants
    if s in ("+1", "1", "+1.0", "1.0"):
        return 0x3FF0000000000000
    if s in ("-1", "-1.0"):
        return 0xBFF0000000000000

    # Parse hexadecimal or standard decimal floating-point representations.
    try:
        val = float.fromhex(s)
    except ValueError:
        val = float(s)
    # Pack as IEEE 754 double (little-endian) and unpack as uint64
    return struct.unpack("<Q", struct.pack("<d", val))[0]


def parse_token_binary32(s: str) -> int:
    """Parse a string token representing a binary32 number to a 32-bit unsigned integer.

    Returns the exact 32-bit IEEE 754 bit-pattern as an integer, or None if the token
    is empty.

    Handles:
      - Signaling NaNs ('snan', '+snan', '-snan')
      - Quiet NaNs ('qnan', 'nan', '+nan', '-nan', '-qnan')
      - Infinities ('inf', '+inf', '-inf', 'infinity', '-infinity')
      - Signed zeros ('+0', '-0', '0.0', '-0.0')
      - Common integer values ('1', '+1', '-1')
      - Hexadecimal floats ('0x1.0p+0') and standard floats
    """
    s = s.strip()
    if not s:
        return None

    # Handle signaling NaNs (quiet bit is 0, payload non-zero)
    if s in ("snan", "+snan"):
        return 0x7FA00000
    if s == "-snan":
        return 0xFFA00000

    # Handle quiet NaNs (quiet bit is 1)
    if s in ("qnan", "+qnan", "nan", "+nan"):
        return 0x7FC00000
    if s in ("-nan", "-qnan"):
        return 0xFFC00000

    # Handle infinities (exponent all 1s, mantissa 0)
    if s in ("inf", "+inf", "infinity", "+infinity"):
        return 0x7F800000
    if s in ("-inf", "-infinity"):
        return 0xFF800000

    # Handle zeros (preserve negative sign bit: 0x80000000)
    if s in ("+0", "0", "+0.0", "0.0"):
        return 0x00000000
    if s in ("-0", "-0.0"):
        return 0x80000000

    # Handle common integer constants
    if s in ("+1", "1", "+1.0", "1.0"):
        return 0x3F800000
    if s in ("-1", "-1.0"):
        return 0xBF800000

    # Parse hexadecimal or standard decimal floating-point representations.
    try:
        val = float.fromhex(s)
    except ValueError:
        val = float(s)
    # Pack as IEEE 754 single float (little-endian) and unpack as uint32
    return struct.unpack("<I", struct.pack("<f", val))[0]


def convert_wc_to_bin(in_stream, out_stream, float_type="binary64") -> int:
    """Convert an input text stream of .wc lines to a raw binary stream.

    Args:
      in_stream: An iterable text stream or file-like object yielding lines.
      out_stream: A writable binary stream (file or sys.stdout.buffer).
      float_type: 'binary64'/'double' (8 bytes) or 'binary32'/'float' (4 bytes).

    Returns:
      The total count of floating-point values successfully converted and written.
    """
    if float_type in ("binary64", "double", "f64"):
        parse_func = parse_token_binary64
        pack_format = "<Q"
    elif float_type in ("binary32", "float", "f32"):
        parse_func = parse_token_binary32
        pack_format = "<I"
    else:
        raise ValueError(f"Unsupported float type: {float_type}")

    count = 0
    buffer = bytearray()
    # Use a 1 MB write buffer to optimize I/O throughput on large files (>1M lines).
    CHUNK_BYTES = 1024 * 1024

    for line in in_stream:
        # Strip comments beginning with '#' and trim whitespace
        line = line.split("#")[0].strip()
        if not line:
            continue

        # Handle comma-separated values (for bivariate functions like pow, hypot)
        parts = line.split(",")
        for part in parts:
            # Take the first token in each part, ignoring any trailing remarks
            tokens = part.split()
            if not tokens:
                continue
            token = tokens[0]
            bit_pattern = parse_func(token)
            if bit_pattern is not None:
                buffer.extend(struct.pack(pack_format, bit_pattern))
                count += 1
                # Flush the buffer when it reaches the chunk size to bound memory use
                if len(buffer) >= CHUNK_BYTES:
                    out_stream.write(buffer)
                    buffer.clear()

    # Flush any remaining bytes in the buffer
    if buffer:
        out_stream.write(buffer)
        buffer.clear()

    return count


def main():
    examples = """Examples:
  Convert sin.wc to binary format (binary64 by default):
    %(prog)s sin.wc sin.bin

  Convert a binary32 worst case file:
    %(prog)s -t binary32 sinf.wc sinf.bin

  Stream from stdin to stdout:
    cat cos.wc | %(prog)s > cos.bin

  Convert and print summary statistics:
    %(prog)s --summary cos.wc cos.bin
"""

    parser = argparse.ArgumentParser(
        description="Convert CORE-MATH .wc files to raw binary IEEE 754 data.",
        epilog=examples,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="-",
        help="Input .wc file path (default: stdin)",
    )
    parser.add_argument(
        "output",
        nargs="?",
        default="-",
        help="Output .bin file path (default: stdout)",
    )
    parser.add_argument(
        "-t",
        "--type",
        choices=["binary64", "double", "binary32", "float"],
        default="binary64",
        help="Target floating point type (default: binary64)",
    )
    parser.add_argument(
        "-s",
        "--summary",
        action="store_true",
        help="Print summary statistics to stderr",
    )

    args = parser.parse_args()

    # Open input stream: defaults to standard input if '-' or unspecified
    if args.input == "-":
        in_stream = sys.stdin
    else:
        in_stream = open(args.input, "r", encoding="utf-8")

    # Open output stream: defaults to standard output if '-' or unspecified
    if args.output == "-":
        out_stream = sys.stdout.buffer
    else:
        out_stream = open(args.output, "wb")

    try:
        count = convert_wc_to_bin(in_stream, out_stream, args.type)
    finally:
        if in_stream is not sys.stdin:
            in_stream.close()
        if out_stream is not sys.stdout.buffer:
            out_stream.close()

    # Print summary if requested or when writing to an explicit output file
    if args.summary or args.output != "-":
        bytes_written = count * (8 if "64" in args.type or "double" in args.type else 4)
        sys.stderr.write(
            f"Converted {count} test cases ({bytes_written} bytes) to {args.output}\n"
        )


if __name__ == "__main__":
    main()

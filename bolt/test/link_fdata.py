#!/usr/bin/env python3

"""
This script reads the input from stdin, extracts all lines starting with
"# FDATA: " (or a given prefix instead of "FDATA"), parses the directives,
replaces symbol names ("#name#") with either symbol values or with offsets from
respective anchor symbols, and prints the resulting file to stdout.
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
import re

parser = argparse.ArgumentParser()
parser.add_argument("input")
parser.add_argument("objfile", help="Object file to extract symbol values from")
parser.add_argument("output")
parser.add_argument("prefix", nargs="?", default="FDATA", help="Custom FDATA prefix")
parser.add_argument(
    "--nmtool",
    default="llvm-nm" if platform.system() == "Windows" else "nm",
    help="Path to nm tool",
)
parser.add_argument("--no-lbr", action="store_true")
parser.add_argument("--no-redefine", action="store_true")

args = parser.parse_args()

# Regexes to extract FDATA lines from input and parse FDATA and pre-aggregated
# profile data
prefix_pat = re.compile(f"^(#|//) {args.prefix}: (.*)")

# FDATA records:
# <is symbol?> <closest elf symbol or DSO name> <relative FROM address>
# <is symbol?> <closest elf symbol or DSO name> <relative TO address>
# <number of mispredictions> <number of branches>
fdata_pat = re.compile(r"([01].*) (?P<mispred>\d+) (?P<exec>\d+)")

# Pre-aggregated profile:
# {T|R|S|E|B|F|f|r} <start> [<end>] [<ft_end>] <count> [<mispred_count>]
# <loc>: [<id>:]<offset>
preagg_pat = re.compile(r"(?P<type>[TRSBFfr]) (?P<offsets_count>.*)")

# No-LBR profile:
# <is symbol?> <closest elf symbol or DSO name> <relative address> <count>
nolbr_pat = re.compile(r"([01].*) (?P<count>\d+)")

# Replacement symbol: #symname#
replace_pat = re.compile(r"#(?P<symname>[^#]+)#")

# Read input and construct the representation of fdata expressions
# as (src_tuple, dst_tuple, mispred_count, exec_count) tuples, where src and dst
# are represented as (is_sym, anchor, offset) tuples
exprs = []
with open(args.input, "r") as f:
    for line in f.readlines():
        prefix_match = prefix_pat.match(line)
        if not prefix_match:
            continue
        profile_line = prefix_match.group(2)
        fdata_match = fdata_pat.match(profile_line)
        preagg_match = preagg_pat.match(profile_line)
        nolbr_match = nolbr_pat.match(profile_line)
        if fdata_match:
            src_dst, mispred, execnt = fdata_match.groups()
            # Split by whitespaces not preceded by a backslash (negative lookbehind)
            chunks = re.split(r"(?<!\\) +", src_dst)
            # Check if the number of records separated by non-escaped whitespace
            # exactly matches the format.
            assert (
                len(chunks) == 6
            ), f"ERROR: wrong format/whitespaces must be escaped:\n{line}"
            exprs.append(("FDATA", (*chunks, mispred, execnt)))
        elif nolbr_match:
            loc, count = nolbr_match.groups()
            # Split by whitespaces not preceded by a backslash (negative lookbehind)
            chunks = re.split(r"(?<!\\) +", loc)
            # Check if the number of records separated by non-escaped whitespace
            # exactly matches the format.
            assert (
                len(chunks) == 3
            ), f"ERROR: wrong format/whitespaces must be escaped:\n{line}"
            exprs.append(("NOLBR", (*chunks, count)))
        elif preagg_match:
            exprs.append(("PREAGG", preagg_match.groups()))
        else:
            exit("ERROR: unexpected input:\n%s" % line)

# Read nm output: <symbol value> <symbol type> <symbol name>
# Ignore .exe on Windows host.
is_llvm_nm = os.path.basename(os.path.realpath(shutil.which(args.nmtool))).startswith(
    "llvm-nm"
)
nm_output = subprocess.run(
    [
        args.nmtool,
        "--defined-only",
        "--special-syms" if is_llvm_nm else "--synthetic",
        args.objfile,
    ],
    text=True,
    capture_output=True,
).stdout
# Populate symbol map
symbols = {}
for symline in nm_output.splitlines():
    symval, _, symname = symline.split(maxsplit=2)
    if symname in symbols and args.no_redefine:
        continue
    symbols[symname] = symval


def evaluate_symbol(issym, anchor, offsym):
    sym_match = replace_pat.match(offsym)
    if not sym_match:
        # No need to evaluate symbol value, return as is
        return f"{issym} {anchor} {offsym}"
    symname = sym_match.group("symname")
    assert symname in symbols, f"ERROR: symbol {symname} is not defined in binary"
    # Evaluate to an absolute offset if issym is false
    if issym == "0":
        return f"{issym} {anchor} {symbols[symname]}"
    # Evaluate symbol against its anchor if issym is true
    assert anchor in symbols, f"ERROR: symbol {anchor} is not defined in binary"
    anchor_value = int(symbols[anchor], 16)
    symbol_value = int(symbols[symname], 16)
    sym_offset = symbol_value - anchor_value
    return f'{issym} {anchor} {format(sym_offset, "x")}'


def replace_symbol(matchobj):
    """
    Expects matchobj to only capture one group which contains the symbol name.
    """
    symname = matchobj.group("symname")
    assert symname in symbols, f"ERROR: symbol {symname} is not defined in binary"
    return symbols[symname]


with open(args.output, "w", newline="\n") as f:
    if args.no_lbr:
        print("no_lbr", file=f)
    for etype, expr in exprs:
        if etype == "FDATA":
            issym1, anchor1, offsym1, issym2, anchor2, offsym2, execnt, mispred = expr
            print(
                evaluate_symbol(issym1, anchor1, offsym1),
                evaluate_symbol(issym2, anchor2, offsym2),
                execnt,
                mispred,
                file=f,
            )
        elif etype == "NOLBR":
            issym, anchor, offsym, count = expr
            print(evaluate_symbol(issym, anchor, offsym), count, file=f)
        elif etype == "PREAGG":
            # Replace all symbols enclosed in ##
            print(expr[0], re.sub(replace_pat, replace_symbol, expr[1]), file=f)
        else:
            exit("ERROR: unhandled expression type:\n%s" % etype)

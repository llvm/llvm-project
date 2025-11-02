#!/usr/bin/env python3
#
# ===-----------------------------------------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===-----------------------------------------------------------------------===#

"""

ClangTidy Alphabetical Order Checker
====================================

Normalize clang-tidy docs with deterministic sorting for linting/tests.

Subcommands:
  - checks-list: Sort entries in docs/clang-tidy/checks/list.rst csv-table.
  - release-notes: Sort key sections in docs/ReleaseNotes.rst and de-duplicate
                   entries in "Changes in existing checks".

Usage:
  clang-tidy-alphabetical-order-check.py <subcommand> [-i <input rst>] [-o <output rst>] [--fix]

Flags:
  -i/--input   Input file.
  -o/--output  Write normalized content here; omit to write to stdout.
  --fix        Rewrite the input file in place. Cannot be combined with -o/--output.
"""

import argparse
import io
import os
import re
import sys
from typing import List, Optional, Sequence, Tuple

DOC_LABEL_RN_RE = re.compile(r":doc:`(?P<label>[^`<]+)\s*(?:<[^>]+>)?`")
DOC_LINE_RE = re.compile(r"^\s*:doc:`(?P<label>[^`<]+?)\s*<[^>]+>`.*$")


def script_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def read_text(path: str) -> List[str]:
    with io.open(path, "r", encoding="utf-8") as f:
        return f.read().splitlines(True)


def write_text(path: str, content: str) -> None:
    with io.open(path, "w", encoding="utf-8", newline="") as f:
        f.write(content)


def normalize_list_rst(lines: List[str]) -> str:
    out: List[str] = []
    i = 0
    n = len(lines)
    while i < n:
        out.append(lines[i])
        if lines[i].lstrip().startswith(".. csv-table::"):
            i += 1
            break
        i += 1

    while i < n and (lines[i].startswith(" ") or lines[i].strip() == ""):
        if DOC_LINE_RE.match(lines[i]):
            break
        out.append(lines[i])
        i += 1

    entries: List[str] = []
    while i < n and lines[i].startswith(" "):
        if DOC_LINE_RE.match(lines[i]):
            entries.append(lines[i])
        else:
            entries.append(lines[i])
        i += 1

    def key_for(line: str):
        m = DOC_LINE_RE.match(line)
        if not m:
            return (1, "")
        return (0, m.group("label"))

    entries_sorted = sorted(entries, key=key_for)
    out.extend(entries_sorted)
    out.extend(lines[i:])

    return "".join(out)


def run_checks_list(inp: Optional[str], out_path: Optional[str], fix: bool) -> int:
    if not inp:
        inp = os.path.normpath(
            os.path.join(
                script_dir(),
                "..",
                "..",
                "docs",
                "clang-tidy",
                "checks",
                "list.rst",
            )
        )
    lines = read_text(inp)
    normalized = normalize_list_rst(lines)
    if fix and out_path:
        sys.stderr.write("error: --fix cannot be used together with --output\n")
        return 2
    if fix:
        original = "".join(lines)
        if original != normalized:
            write_text(inp, normalized)
        return 0
    if out_path:
        write_text(out_path, normalized)
        return 0
    sys.stdout.write(normalized)
    return 0


def find_heading(lines: Sequence[str], title: str) -> Optional[int]:
    for i in range(len(lines) - 1):
        if lines[i].rstrip("\n") == title:
            underline = lines[i + 1].rstrip("\n")
            if underline and set(underline) == {"^"} and len(underline) >= len(title):
                return i
    return None


def extract_label(text: str) -> str:
    m = DOC_LABEL_RN_RE.search(text)
    return m.group("label") if m else text


def is_bullet_start(line: str) -> bool:
    return line.startswith("- ")


def collect_bullet_blocks(
    lines: Sequence[str], start: int, end: int
) -> Tuple[List[str], List[Tuple[str, List[str]]], List[str]]:
    i = start
    n = end
    first_bullet = i
    while first_bullet < n and not is_bullet_start(lines[first_bullet]):
        first_bullet += 1
    prefix = list(lines[i:first_bullet])

    blocks: List[Tuple[str, List[str]]] = []
    i = first_bullet
    while i < n:
        if not is_bullet_start(lines[i]):
            break
        bstart = i
        i += 1
        while i < n and not is_bullet_start(lines[i]):
            if (
                i + 1 < n
                and set(lines[i + 1].rstrip("\n")) == {"^"}
                and lines[i].strip()
            ):
                break
            i += 1
        block = list(lines[bstart:i])
        key = extract_label(block[0])
        blocks.append((key, block))

    suffix = list(lines[i:n])
    return prefix, blocks, suffix


def sort_and_dedup_blocks(
    blocks: List[Tuple[str, List[str]]], dedup: bool = False
) -> List[List[str]]:
    seen = set()
    filtered: List[Tuple[str, List[str]]] = []
    for key, block in blocks:
        if dedup:
            if key in seen:
                continue
            seen.add(key)
        filtered.append((key, block))
    filtered.sort(key=lambda kb: kb[0])
    return [b for _, b in filtered]


def normalize_release_notes(lines: List[str]) -> str:
    sections = [
        ("New checks", False),
        ("New check aliases", False),
        ("Changes in existing checks", True),
    ]

    out = list(lines)

    for idx in range(len(sections) - 1, -1, -1):
        title, dedup = sections[idx]
        h_start = find_heading(out, title)

        if h_start is None:
            continue

        sec_start = h_start + 2

        if idx + 1 < len(sections):
            next_title = sections[idx + 1][0]
            h_end = find_heading(out, next_title)
            if h_end is None:
                h_end = sec_start
                while h_end + 1 < len(out):
                    if out[h_end].strip() and set(out[h_end + 1].rstrip("\n")) == {"^"}:
                        break
                    h_end += 1
            sec_end = h_end
        else:
            h_end = sec_start
            while h_end + 1 < len(out):
                if out[h_end].strip() and set(out[h_end + 1].rstrip("\n")) == {"^"}:
                    break
                h_end += 1
            sec_end = h_end

        prefix, blocks, suffix = collect_bullet_blocks(out, sec_start, sec_end)
        sorted_blocks = sort_and_dedup_blocks(blocks, dedup=dedup)

        new_section: List[str] = []
        new_section.extend(prefix)
        for i_b, b in enumerate(sorted_blocks):
            if i_b > 0 and (
                    not new_section or (new_section and new_section[-1].strip() != "")
            ):
                new_section.append("\n")
            new_section.extend(b)
        new_section.extend(suffix)

        out = out[:sec_start] + new_section + out[sec_end:]

    return "".join(out)


def run_release_notes(inp: Optional[str], out_path: Optional[str], fix: bool) -> int:
    if not inp:
        inp = os.path.normpath(
            os.path.join(script_dir(), "..", "..", "docs", "ReleaseNotes.rst")
        )
    lines = read_text(inp)
    normalized = normalize_release_notes(lines)
    if fix and out_path:
        sys.stderr.write("error: --fix cannot be used together with --output\n")
        return 2
    if fix:
        original = "".join(lines)
        if original != normalized:
            write_text(inp, normalized)
        return 0
    if out_path:
        write_text(out_path, normalized)
        return 0
    sys.stdout.write(normalized)
    return 0


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_checks = sub.add_parser(
        "checks-list", help="normalize clang-tidy checks list.rst"
    )
    ap_checks.add_argument("-i", "--input", dest="inp", default=None)
    ap_checks.add_argument("-o", "--output", dest="out", default=None)
    ap_checks.add_argument(
        "--fix", action="store_true", help="rewrite the input file in place"
    )

    ap_rn = sub.add_parser("release-notes", help="normalize ReleaseNotes.rst sections")
    ap_rn.add_argument("-i", "--input", dest="inp", default=None)
    ap_rn.add_argument("-o", "--output", dest="out", default=None)
    ap_rn.add_argument(
        "--fix", action="store_true", help="rewrite the input file in place"
    )

    args = ap.parse_args(argv)

    if args.cmd == "checks-list":
        return run_checks_list(args.inp, args.out, args.fix)
    if args.cmd == "release-notes":
        return run_release_notes(args.inp, args.out, args.fix)

    ap.error("unknown command")


if __name__ == "__main__":
    main(sys.argv[1:])

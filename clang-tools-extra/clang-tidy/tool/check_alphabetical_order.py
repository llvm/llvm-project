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

Clang-Tidy Alphabetical Order Checker
=====================================

Normalize Clang-Tidy documentation with deterministic sorting for linting/tests.

Behavior:
- Sort entries in docs/clang-tidy/checks/list.rst csv-table.
- Sort key sections in docs/ReleaseNotes.rst.
- Detect duplicated entries in 'Changes in existing checks'.

Flags:
  -o/--output  Write normalized content to this path instead of updating docs.
"""

import argparse
import io
import os
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple, NamedTuple
from operator import itemgetter

# Matches a :doc:`label <path>` or :doc:`label` reference anywhere in text and
# captures the label. Used to sort bullet items alphabetically in ReleaseNotes
# items by their label.
DOC_LABEL_RN_RE = re.compile(r":doc:`(?P<label>[^`<]+)\s*(?:<[^>]+>)?`")

# Matches a single csv-table row line in list.rst that begins with a :doc:
# reference, capturing the label. Used to extract the sort key per row.
DOC_LINE_RE = re.compile(r"^\s*:doc:`(?P<label>[^`<]+?)\s*<[^>]+>`.*$")


EXTRA_DIR = os.path.join(os.path.dirname(__file__), "../..")
DOCS_DIR = os.path.join(EXTRA_DIR, "docs")
CLANG_TIDY_DOCS_DIR = os.path.join(DOCS_DIR, "clang-tidy")
CHECKS_DOCS_DIR = os.path.join(CLANG_TIDY_DOCS_DIR, "checks")
LIST_DOC = os.path.join(CHECKS_DOCS_DIR, "list.rst")
RELEASE_NOTES_DOC = os.path.join(DOCS_DIR, "ReleaseNotes.rst")


CheckLabel = str
Lines = List[str]
BulletBlock = List[str]
BulletItem = Tuple[CheckLabel, BulletBlock]
BulletStart = int


class BulletBlocks(NamedTuple):
    """Structured result of parsing a bullet-list section.

    - prefix: lines before the first bullet within the section range.
    - blocks: list of (label, block-lines) pairs for each bullet block.
    - suffix: lines after the last bullet within the section range.
    """
    prefix: Lines
    blocks: List[BulletItem]
    suffix: Lines

class ScannedBlocks(NamedTuple):
    """Result of scanning bullet blocks within a section range.

    - blocks_with_pos: list of (start_index, block_lines) for each bullet block.
    - next_index: index where scanning stopped; start of the suffix region.
    """
    blocks_with_pos: List[Tuple[BulletStart, BulletBlock]]
    next_index: int


def _scan_bullet_blocks(
    lines: Sequence[str], start: int, end: int
) -> ScannedBlocks:
    """Scan consecutive bullet blocks and return (blocks_with_pos, next_index).

    Each entry in blocks_with_pos is a tuple of (start_index, block_lines).
    next_index is the index where scanning stopped (start of suffix).
    """
    i = start
    n = end
    blocks_with_pos: List[Tuple[BulletStart, BulletBlock]] = []
    while i < n:
        if not _is_bullet_start(lines[i]):
            break
        bstart = i
        i += 1
        while i < n and not _is_bullet_start(lines[i]):
            if (
                i + 1 < n
                and set(lines[i + 1].rstrip("\n")) == {"^"}
                and lines[i].strip()
            ):
                break
            i += 1
        block: BulletBlock = list(lines[bstart:i])
        blocks_with_pos.append((bstart, block))
    return ScannedBlocks(blocks_with_pos, i)


def read_text(path: str) -> List[str]:
    with io.open(path, "r", encoding="utf-8") as f:
        return f.read().splitlines(True)


def write_text(path: str, content: str) -> None:
    with io.open(path, "w", encoding="utf-8", newline="") as f:
        f.write(content)


def _normalize_list_rst_lines(lines: Sequence[str]) -> List[str]:
    """Return normalized content of checks list.rst as a list of lines."""
    out: List[str] = []
    i = 0
    n = len(lines)

    def check_name(line: str):
        m = DOC_LINE_RE.match(line)
        if not m:
            return (1, "")
        return (0, m.group("label"))

    while i < n:
        line = lines[i]
        if line.lstrip().startswith(".. csv-table::"):
            out.append(line)
            i += 1

            while i < n and (lines[i].startswith(" ") or lines[i].strip() == ""):
                if DOC_LINE_RE.match(lines[i]):
                    break
                out.append(lines[i])
                i += 1

            entries: List[str] = []
            while i < n and lines[i].startswith(" "):
                entries.append(lines[i])
                i += 1

            entries_sorted = sorted(entries, key=check_name)
            out.extend(entries_sorted)
            continue

        out.append(line)
        i += 1

    return out


def normalize_list_rst(data: str) -> str:
    """Normalize list.rst content and return a string."""
    lines = data.splitlines(True)
    return "".join(_normalize_list_rst_lines(lines))


def find_heading(lines: Sequence[str], title: str) -> Optional[int]:
    """Find heading start index for a section underlined with ^ characters.

    The function looks for a line equal to `title` followed by a line that
    consists solely of ^, which matches the ReleaseNotes style for subsection
    headings used here.

    Returns index of the title line, or None if not found.
    """
    for i in range(len(lines) - 1):
        if lines[i].rstrip("\n") == title:
            underline = lines[i + 1].rstrip("\n")
            if underline and set(underline) == {"^"} and len(underline) == len(title):
                return i
    return None


def extract_label(text: str) -> str:
    m = DOC_LABEL_RN_RE.search(text)
    return m.group("label") if m else text


def _is_bullet_start(line: str) -> bool:
    return line.startswith("- ")


def _parse_bullet_blocks(
    lines: Sequence[str], start: int, end: int
) -> BulletBlocks:
    i = start
    n = end
    first_bullet = i
    while first_bullet < n and not _is_bullet_start(lines[first_bullet]):
        first_bullet += 1
    prefix: Lines = list(lines[i:first_bullet])

    blocks: List[BulletItem] = []
    res = _scan_bullet_blocks(lines, first_bullet, n)
    for _, block in res.blocks_with_pos:
        key: CheckLabel = extract_label(block[0])
        blocks.append((key, block))

    suffix: Lines = list(lines[res.next_index:n])
    return BulletBlocks(prefix, blocks, suffix)


def sort_blocks(blocks: List[BulletItem]) -> List[BulletBlock]:
    """Return blocks sorted deterministically by their extracted label.

    Duplicates are preserved; merging is left to authors to handle manually.
    """
    return list(map(itemgetter(1), sorted(blocks, key=itemgetter(0))))


def find_duplicate_entries(
    lines: Sequence[str], title: str
) -> List[Tuple[str, List[Tuple[int, List[str]]]]]:
    """Return detailed duplicate info as (key, [(start_idx, block_lines), ...]).

    start_idx is the 0-based index of the first line of the bullet block in
    the original lines list. Only keys with more than one occurrence are
    returned, and occurrences are listed in the order they appear.
    """
    bounds = _find_section_bounds(lines, title, None)
    if bounds is None:
        return []
    _, sec_start, sec_end = bounds

    i = sec_start
    n = sec_end

    while i < n and not _is_bullet_start(lines[i]):
        i += 1

    blocks_with_pos: List[Tuple[str, int, List[str]]] = []
    res = _scan_bullet_blocks(lines, i, n)
    for bstart, block in res.blocks_with_pos:
        key = extract_label(block[0])
        blocks_with_pos.append((key, bstart, block))

    grouped: Dict[str, List[Tuple[int, List[str]]]] = {}
    for key, start, block in blocks_with_pos:
        grouped.setdefault(key, []).append((start, block))

    result: List[Tuple[str, List[Tuple[int, List[str]]]]] = []
    for key, occs in grouped.items():
        if len(occs) > 1:
            result.append((key, occs))

    result.sort(key=itemgetter(0))
    return result


def _find_section_bounds(
    lines: Sequence[str], title: str, next_title: Optional[str]
) -> Optional[Tuple[int, int, int]]:
    """Return (h_start, sec_start, sec_end) for section `title`.

    - h_start: index of the section title line
    - sec_start: index of the first content line after underline
    - sec_end: index of the first line of the next section title (or end)
    """
    h_start = find_heading(lines, title)
    if h_start is None:
        return None

    sec_start = h_start + 2

    # Determine end of section either from next_title or by scanning.
    if next_title is not None:
        h_end = find_heading(lines, next_title)
        if h_end is None:
            # Scan forward to the next heading-like underline.
            h_end = sec_start
            while h_end + 1 < len(lines):
                if lines[h_end].strip() and set(lines[h_end + 1].rstrip("\n")) == {"^"}:
                    break
                h_end += 1
        sec_end = h_end
    else:
        # Scan to end or until a heading underline is found.
        h_end = sec_start
        while h_end + 1 < len(lines):
            if lines[h_end].strip() and set(lines[h_end + 1].rstrip("\n")) == {"^"}:
                break
            h_end += 1
        sec_end = h_end

    return h_start, sec_start, sec_end


def _normalize_release_notes_section(
    lines: Sequence[str], title: str, next_title: Optional[str]
) -> List[str]:
    """Normalize a single release-notes section and return updated lines."""
    bounds = _find_section_bounds(lines, title, next_title)
    if bounds is None:
        return list(lines)
    _, sec_start, sec_end = bounds

    prefix, blocks, suffix = _parse_bullet_blocks(lines, sec_start, sec_end)
    sorted_blocks = sort_blocks(blocks)

    new_section: List[str] = []
    new_section.extend(prefix)
    for i_b, b in enumerate(sorted_blocks):
        if i_b > 0 and (
            not new_section or (new_section and new_section[-1].strip() != "")
        ):
            new_section.append("\n")
        new_section.extend(b)
    new_section.extend(suffix)

    return list(lines[:sec_start]) + new_section + list(lines[sec_end:])


def normalize_release_notes(lines: Sequence[str]) -> str:
    sections = ["New checks", "New check aliases", "Changes in existing checks"]

    out = list(lines)

    for idx in range(len(sections) - 1, -1, -1):
        title = sections[idx]
        next_title = sections[idx + 1] if idx + 1 < len(sections) else None
        out = _normalize_release_notes_section(out, title, next_title)

    return "".join(out)


def _emit_duplicate_report(lines: Sequence[str], title: str) -> Optional[str]:
    dups_detail = find_duplicate_entries(lines, title)
    if not dups_detail:
        return None
    out: List[str] = []
    out.append(f"Error: Duplicate entries in '{title}':\n")
    for key, occs in dups_detail:
        out.append(f"\n-- Duplicate: {key}\n")
        for start_idx, block in occs:
            out.append(f"- At line {start_idx + 1}:\n")
            out.append("".join(block))
            if not (block and block[-1].endswith("\n")):
                out.append("\n")
    return "".join(out)


def process_release_notes(out_path: str, rn_doc: str) -> int:
    lines = read_text(rn_doc)
    normalized = normalize_release_notes(lines)
    write_text(out_path, normalized)

    # Prefer reporting ordering issues first; let diff fail the test.
    if "".join(lines) != normalized:
        sys.stderr.write(
            "Note: 'ReleaseNotes.rst' is not normalized; Please fix ordering first.\n"
        )
        return 0

    # Ordering is clean then enforce duplicates.
    report = _emit_duplicate_report(lines, "Changes in existing checks")
    if report:
        sys.stderr.write(report)
        return 3
    return 0


def process_checks_list(out_path: str, list_doc: str) -> int:
    lines = read_text(list_doc)
    normalized = normalize_list_rst("".join(lines))
    write_text(out_path, normalized)
    return 0


def main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", dest="out", default=None)
    args = ap.parse_args(argv)

    list_doc, rn_doc = (os.path.normpath(LIST_DOC), os.path.normpath(RELEASE_NOTES_DOC))

    if args.out:
        out_path = args.out
        out_lower = os.path.basename(out_path).lower()
        if "release" in out_lower:
            return process_release_notes(out_path, rn_doc)
        else:
            return process_checks_list(out_path, list_doc)

    process_checks_list(list_doc, list_doc)
    return process_release_notes(rn_doc, rn_doc)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

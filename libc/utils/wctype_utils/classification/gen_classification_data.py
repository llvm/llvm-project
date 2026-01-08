# ===- Generate classification tables for wctype utils -----*- python -*----==#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ==------------------------------------------------------------------------==#


from enum import IntFlag
from dataclasses import dataclass
from collections import defaultdict
from sys import argv


# WARNING: If you modify this enum, you must update the generated C++ enum
# in generate_code as well
class PropertyFlag(IntFlag):
    UPPER = 1 << 0
    LOWER = 1 << 1
    ALPHA = 1 << 2
    SPACE = 1 << 3
    PRINT = 1 << 4
    BLANK = 1 << 5
    CNTRL = 1 << 6
    PUNCT = 1 << 7


@dataclass
class UnicodeEntry:
    codepoint: int
    name: str
    category: str


def read_unicode_data(filename: str) -> list[UnicodeEntry]:
    """Reads Unicode data from file and returns list of entries."""
    entries: list[UnicodeEntry] = []

    try:
        with open(filename, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()

                if not line or line.startswith("#"):
                    continue

                fields = line.split(";")

                if len(fields) < 3:
                    continue

                codepoint_str = fields[0].strip()
                name = fields[1].strip()
                category = fields[2].strip()

                codepoint = int(codepoint_str, 16)

                entries.append(UnicodeEntry(codepoint, name, category))

    except FileNotFoundError:
        raise RuntimeError(f"Cannot open file: {filename}")

    return entries


from dataclasses import dataclass

# Non-whitespace spaces in C.UTF-8
NON_WHITESPACE_SPACES = {0x00A0, 0x2007, 0x202F}

ASCII_DIGITS = {0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39}


def handle_ranges(
    properties: defaultdict[int, int], entries: list[UnicodeEntry]
) -> None:
    """Handles Unicode ranges defined by <First> and <Last>."""
    range_start: int | None = None
    range_props: int | None = None

    for entry in entries:
        if ", First>" in entry.name:
            range_start = entry.codepoint
            range_props = properties[entry.codepoint]
        elif ", Last>" in entry.name and range_start and range_props:
            for cp in range(range_start, entry.codepoint + 1):
                properties[cp] = range_props
            range_start = None
            range_props = None


def get_props(entry: UnicodeEntry) -> int:
    """Creates the property flag for a given UnicodeEntry."""
    codepoint = entry.codepoint
    category = entry.category
    props = 0

    match category[0]:
        case "L":
            props |= PropertyFlag.ALPHA
            if category in ("Lu", "Lt"):
                props |= PropertyFlag.UPPER
            elif category == "Ll":
                props |= PropertyFlag.LOWER

        case "N":
            # In C.UTF8, non-ASCII digits/letter-numbers are alpha
            if category in ("Nd", "Nl") and codepoint not in ASCII_DIGITS:
                props |= PropertyFlag.ALPHA

        case "P" | "S":
            # Symbols are considered punctuation in C.UTF8
            props |= PropertyFlag.PUNCT

        case "Z":
            if codepoint not in NON_WHITESPACE_SPACES:
                props |= PropertyFlag.SPACE
                if category == "Zs":
                    props |= PropertyFlag.BLANK

        case "C":
            if category == "Cc":
                props |= PropertyFlag.CNTRL

    # Print = all except control, unassigned, surrogate, format
    if category not in ("Cc", "Cs", "Cn", "Cf"):
        props |= PropertyFlag.PRINT

    return props


def handle_special_cases(properties: defaultdict[int, int]) -> None:
    """Handles special cases not parseable from UnicodeData.txt."""
    # ASCII whitespace characters
    properties[0x0020] |= PropertyFlag.SPACE  # SPACE
    properties[0x0009] |= PropertyFlag.SPACE  # TAB
    properties[0x000A] |= PropertyFlag.SPACE  # LINE FEED
    properties[0x000D] |= PropertyFlag.SPACE  # CARRIAGE RETURN
    properties[0x000B] |= PropertyFlag.SPACE  # VERTICAL TAB
    properties[0x000C] |= PropertyFlag.SPACE  # FORM FEED

    # Blank
    properties[0x0020] |= PropertyFlag.BLANK  # SPACE
    properties[0x0009] |= PropertyFlag.BLANK  # TAB


def parse_unicode_data(entries: list[UnicodeEntry]) -> defaultdict[int, int]:
    """Returns codepoint -> property flag mappings."""
    properties: defaultdict[int, int] = defaultdict(int)

    for entry in entries:
        codepoint = entry.codepoint

        # Skip surrogate pairs
        if 0xD800 <= codepoint <= 0xDFFF:
            continue

        properties[codepoint] = get_props(entry)

    handle_ranges(properties, entries)
    handle_special_cases(properties)

    return properties


@dataclass
class StagedLookupTable:
    level1: list[int]  # Maps codepoint >> 8 to level2 offset
    level2: list[int]  # Actual properties


def build_lookup_tables(properties: defaultdict[int, int]) -> StagedLookupTable:
    """Builds two-level lookup tables."""
    UNICODE_MAX = 0x110000
    BLOCK_SIZE = 256
    NUM_BLOCKS = UNICODE_MAX // BLOCK_SIZE

    # Maps block content -> block index in level2
    blocks: defaultdict[tuple[int, ...], int] = defaultdict(int)
    level1: list[int] = []
    level2: list[int] = []

    for block_num in range(NUM_BLOCKS):
        block_content = tuple(
            properties.get((block_num << 8) | offset, 0) for offset in range(BLOCK_SIZE)
        )

        if block_content in blocks:
            # Reuse existing block
            level1.append(blocks[block_content])
        else:
            # New block - add to level2
            block_index = len(level2)
            blocks[block_content] = block_index

            level2.extend(block_content)
            level1.append(block_index)

    print("Table statistics:")
    print(f"  Level 1 entries: {len(level1)}")
    print(f"  Level 2 entries: {len(level2)}")
    print(f"  Size: {len(level1) * 2 + len(level2)} bytes")

    return StagedLookupTable(level1, level2)


def generate_code(lookup_table: StagedLookupTable, llvm_project_root_path: str) -> None:
    """Generates C++ header with lookup tables."""
    level1 = lookup_table.level1
    level2 = lookup_table.level2

    with open(
        f"{llvm_project_root_path}/libc/src/__support/wctype/wctype_classification_utils.h",
        "w",
    ) as f:
        f.write(
            f"""//===-- Utils for wctype classification functions ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// DO NOT EDIT MANUALLY.
// This file is generated by libc/utils/wctype_utils scripts.

#ifndef LLVM_LIBC_SRC___SUPPORT_WCTYPE_WCTYPE_CLASSIFICATION_UTILS_H
#define LLVM_LIBC_SRC___SUPPORT_WCTYPE_WCTYPE_CLASSIFICATION_UTILS_H

#include "hdr/stdint_proxy.h" 
#include "hdr/types/wchar_t.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"
#include "src/__support/libc_assert.h"
#include "src/__support/CPP/limits.h"

namespace LIBC_NAMESPACE_DECL {{

// Property flags for Unicode categories
enum PropertyFlag : uint8_t {{
  UPPER = 1 << 0,
  LOWER = 1 << 1,
  ALPHA = 1 << 2,
  SPACE = 1 << 3,
  PRINT = 1 << 4,
  BLANK = 1 << 5,
  CNTRL = 1 << 6,
  PUNCT = 1 << 7,
}};

static_assert({len(level1)} <= cpp::numeric_limits<unsigned short>::max());
static_assert({len(level2)} <= cpp::numeric_limits<unsigned short>::max());

LIBC_INLINE_VAR constexpr uint16_t LEVEL1_SIZE = {len(level1)};
LIBC_INLINE_VAR constexpr uint16_t LEVEL2_SIZE = {len(level2)};

// Level 1 table: indexed by (codepoint >> 8), stores level2 block offsets
LIBC_INLINE_VAR constexpr uint16_t level1[LEVEL1_SIZE] = {{
"""
        )
        for i in range(0, len(level1), 11):
            f.write("  ")
            for j in range(i, min(i + 11, len(level1))):
                f.write(f"{level1[j]:7d}")
                if j + 1 < len(level1):
                    f.write(",")
            f.write("\n")
        f.write(
            f"""}};

// Level 2 table: blocks of 256 property flags
LIBC_INLINE_VAR constexpr uint8_t level2[LEVEL2_SIZE] = {{
"""
        )
        for i in range(0, len(level2), 11):
            f.write("  ")
            for j in range(i, min(i + 11, len(level2))):
                f.write(f"0x{level2[j]:02x}")
                if j + 1 < len(level2):
                    f.write(", ")
            f.write("\n")
        f.write(
            f"""}};

// Returns the Unicode property flag for a given wide character.
LIBC_INLINE constexpr uint8_t lookup_properties(const wchar_t wc) {{
  // Out of Unicode range
  if (static_cast<uint32_t>(wc) > 0x10FFFF) {{
    return 0;
  }}

  uint16_t l1_idx = static_cast<uint16_t>(wc >> 8);
  LIBC_ASSERT(l1_idx < LEVEL1_SIZE);

  uint16_t l2_offset = level1[l1_idx];
  uint16_t l2_idx = l2_offset + (wc & 0xFF);
  LIBC_ASSERT(l2_idx < LEVEL2_SIZE);

  return level2[l2_idx];
}}

}} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_WCTYPE_WCTYPE_CLASSIFICATION_UTILS_H

"""
        )

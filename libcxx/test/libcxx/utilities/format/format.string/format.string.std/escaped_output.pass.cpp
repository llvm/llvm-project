//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// UNSUPPORTED: GCC-ALWAYS_INLINE-FIXME

// <format>

// Tests the properties of the Unicode escaped output table.
// The libc++ algorithm has size and speed optimizations based on the properties
// of Unicode. This means updating the Unicode tables has a likilihood of
// breaking test. This is an assert; it requires validating whether the
// assumptions of the size and speed optimizations are still valid.

#include <algorithm>
#include <numeric>
#include <format>
#include <cassert>

// Contains the entries for [format.string.escaped]/2.2.1.2.1
//   CE is a Unicode encoding and C corresponds to a UCS scalar value whose
//   Unicode property General_Category has a value in the groups Separator (Z)
//   or Other (C), as described by table 12 of UAX #44
//
// Separator (Z) consists of General_Category
// - Zs Space_Separator,
// - Zl Line_Separator,
// - Zp Paragraph_Separator.
//
// Other (C) consists of General_Category
// - Cc Control,
// - Cf Format,
// - Cs Surrogate,
// - Co Private_Use,
// - Cn Unassigned.
inline constexpr int Zs = 17;
inline constexpr int Zl = 1;
inline constexpr int Zp = 1;
inline constexpr int Z  = Zs + Zl + Zp;

inline constexpr int Cc = 65;
inline constexpr int Cf = 170;
inline constexpr int Cs = 2'048;
inline constexpr int Co = 137'468;
inline constexpr int Cn = 824'718;
inline constexpr int C  = Cc + Cf + Cs + Co + Cn;

// This is the final part of the Unicode properties table:
//
// 31350..323AF  ; Lo # [4192] CJK UNIFIED IDEOGRAPH-31350..CJK UNIFIED IDEOGRAPH-323AF
// 323B0..E0000  ; Cn # [711761] <reserved-323B0>..<reserved-E0000>
// E0001         ; Cf #       LANGUAGE TAG
// E0002..E001F  ; Cn #  [30] <reserved-E0002>..<reserved-E001F>
// E0020..E007F  ; Cf #  [96] TAG SPACE..CANCEL TAG
// E0080..E00FF  ; Cn # [128] <reserved-E0080>..<reserved-E00FF>
// E0100..E01EF  ; Mn # [240] VARIATION SELECTOR-17..VARIATION SELECTOR-256
// E01F0..EFFFF  ; Cn # [65040] <reserved-E01F0>..<noncharacter-EFFFF>
// F0000..FFFFD  ; Co # [65534] <private-use-F0000>..<private-use-FFFFD>
// FFFFE..FFFFF  ; Cn #   [2] <noncharacter-FFFFE>..<noncharacter-FFFFF>
// 100000..10FFFD; Co # [65534] <private-use-100000>..<private-use-10FFFD>
// 10FFFE..10FFFF; Cn #   [2] <noncharacter-10FFFE>..<noncharacter-10FFFF>
//
// It can be observed all entries in the range 323B0..10FFFF are in the
// categories Cf, Co, Cn, except a small range with the property Mn.
// In order to reduce the size of the table only the entires in the range
// [0000, 323B0) are stored in the table. The entries in the range
// [323B0, 10FFFF] use a hand-crafted algorithm.
//
// This means a number of entries are omitted
inline constexpr int excluded = ((0x10FFFF - 0x323B0) + 1) - 240;

inline constexpr int entries = Z + C - excluded;

static constexpr int count_entries() {
  return std::transform_reduce(
      std::begin(std::__escaped_output_table::__entries),
      std::end(std::__escaped_output_table::__entries),
      0,
      std::plus{},
      [](auto entry) { return 1 + static_cast<int>(entry & 0x3fffu); });
}
static_assert(count_entries() == entries);

int main(int, char**) {
  for (char32_t c = 0x31350; c <= 0x323AF; ++c) // 31350..323AF  ; Lo # [4192]
    assert(std::__escaped_output_table::__needs_escape(c) == false);

  for (char32_t c = 0x323B0; c <= 0xE00FF; ++c) // 323B0..E00FF ; C
    assert(std::__escaped_output_table::__needs_escape(c) == true);

  for (char32_t c = 0xE0100; c <= 0xE01EF; ++c) // E0100..E01EF  ; Mn # [240]
    assert(std::__escaped_output_table::__needs_escape(c) == false);

  for (char32_t c = 0xE01F0; c <= 0x10FFFF; ++c) // E01F0..10FFFF; C
    assert(std::__escaped_output_table::__needs_escape(c) == true);

  return 0;
}

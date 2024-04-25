//===-- runtime/utf.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "utf.h"

namespace Fortran::runtime {

// clang-format off
RT_OFFLOAD_VAR_GROUP_BEGIN
const RT_CONST_VAR_ATTRS std::uint8_t UTF8FirstByteTable[256]{
  /* 00 - 7F:  7 bit payload in single byte */
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  /* 80 - BF: invalid first byte, valid later byte */
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  /* C0 - DF: 11 bit payload */
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
  /* E0 - EF: 16 bit payload */
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
  /* F0 - F7: 21 bit payload */ 4, 4, 4, 4, 4, 4, 4, 4,
  /* F8 - FB: 26 bit payload */ 5, 5, 5, 5,
  /* FC - FD: 31 bit payload */ 6, 6,
  /* FE:      32 bit payload */ 7,
  /* FF:      invalid */ 0
};
RT_OFFLOAD_VAR_GROUP_END
// clang-format on

RT_OFFLOAD_API_GROUP_BEGIN
// Non-minimal encodings are accepted.
Fortran::common::optional<char32_t> DecodeUTF8(const char *p0) {
  const std::uint8_t *p{reinterpret_cast<const std::uint8_t *>(p0)};
  std::size_t bytes{MeasureUTF8Bytes(*p0)};
  if (bytes == 1) {
    return char32_t{*p};
  } else if (bytes > 1) {
    std::uint64_t result{char32_t{*p} & (0x7f >> bytes)};
    for (std::size_t j{1}; j < bytes; ++j) {
      std::uint8_t next{p[j]};
      if (next < 0x80 || next > 0xbf) {
        return Fortran::common::nullopt;
      }
      result = (result << 6) | (next & 0x3f);
    }
    if (result <= 0xffffffff) {
      return static_cast<char32_t>(result);
    }
  }
  return Fortran::common::nullopt;
}

std::size_t EncodeUTF8(char *p0, char32_t ucs) {
  std::uint8_t *p{reinterpret_cast<std::uint8_t *>(p0)};
  if (ucs <= 0x7f) {
    p[0] = ucs;
    return 1;
  } else if (ucs <= 0x7ff) {
    p[0] = 0xc0 | (ucs >> 6);
    p[1] = 0x80 | (ucs & 0x3f);
    return 2;
  } else if (ucs <= 0xffff) {
    p[0] = 0xe0 | (ucs >> 12);
    p[1] = 0x80 | ((ucs >> 6) & 0x3f);
    p[2] = 0x80 | (ucs & 0x3f);
    return 3;
  } else if (ucs <= 0x1fffff) {
    p[0] = 0xf0 | (ucs >> 18);
    p[1] = 0x80 | ((ucs >> 12) & 0x3f);
    p[2] = 0x80 | ((ucs >> 6) & 0x3f);
    p[3] = 0x80 | (ucs & 0x3f);
    return 4;
  } else if (ucs <= 0x3ffffff) {
    p[0] = 0xf8 | (ucs >> 24);
    p[1] = 0x80 | ((ucs >> 18) & 0x3f);
    p[2] = 0x80 | ((ucs >> 12) & 0x3f);
    p[3] = 0x80 | ((ucs >> 6) & 0x3f);
    p[4] = 0x80 | (ucs & 0x3f);
    return 5;
  } else if (ucs <= 0x7ffffff) {
    p[0] = 0xf8 | (ucs >> 30);
    p[1] = 0x80 | ((ucs >> 24) & 0x3f);
    p[2] = 0x80 | ((ucs >> 18) & 0x3f);
    p[3] = 0x80 | ((ucs >> 12) & 0x3f);
    p[4] = 0x80 | ((ucs >> 6) & 0x3f);
    p[5] = 0x80 | (ucs & 0x3f);
    return 6;
  } else {
    p[0] = 0xfe;
    p[1] = 0x80 | ((ucs >> 30) & 0x3f);
    p[2] = 0x80 | ((ucs >> 24) & 0x3f);
    p[3] = 0x80 | ((ucs >> 18) & 0x3f);
    p[4] = 0x80 | ((ucs >> 12) & 0x3f);
    p[5] = 0x80 | ((ucs >> 6) & 0x3f);
    p[6] = 0x80 | (ucs & 0x3f);
    return 7;
  }
}
RT_OFFLOAD_API_GROUP_END

} // namespace Fortran::runtime

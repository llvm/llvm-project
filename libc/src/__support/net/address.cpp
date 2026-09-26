//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements helper functions for parsing and formatting network
/// addresses.
///
//===----------------------------------------------------------------------===//

#include "src/__support/net/address.h"
#include "hdr/inet-address-macros.h"
#include "hdr/stdint_proxy.h"
#include "hdr/types/in_addr_t.h"
#include "hdr/types/struct_in6_addr.h"
#include "hdr/types/struct_in_addr.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/common.h"
#include "src/__support/ctype_utils.h"
#include "src/__support/endian_internal.h"
#include "src/__support/libc_assert.h"
#include "src/__support/str_to_integer.h"
#include "src/string/memory_utils/inline_memcpy.h"

namespace LIBC_NAMESPACE_DECL {

namespace net {

[[nodiscard]] bool str_to_ipv4(cpp::string_view src, struct in_addr &dst) {
  uint8_t bytes[4];
  size_t idx = 0;
  uint32_t current_val = 0;
  size_t digits_in_octet = 0;

  for (char c : src) {
    if (internal::isdigit(c)) {
      // Reject octals and leading zeros
      if (digits_in_octet > 0 && current_val == 0)
        return false;

      current_val = current_val * 10 + internal::b36_char_to_int(c);
      if (current_val > 255)
        return false;

      ++digits_in_octet;
    } else if (c == '.') {
      if (digits_in_octet == 0 || idx == 3)
        return false; // Empty part or too many dots

      bytes[idx++] = static_cast<uint8_t>(current_val);
      current_val = 0;
      digits_in_octet = 0;
    } else {
      return false;
    }
  }

  if (idx != 3 || digits_in_octet == 0)
    return 0;

  bytes[3] = static_cast<uint8_t>(current_val);
  inline_memcpy(&dst.s_addr, bytes, 4);
  return true;
}

namespace {

LIBC_INLINE constexpr bool is_hex_char(char c) {
  return internal::isalnum(c) && internal::b36_char_to_int(c) < 16;
}

} // anonymous namespace

[[nodiscard]] bool str_to_ipv6(cpp::string_view src, struct in6_addr &dst) {
  if (src.empty())
    return false;

  uint8_t bytes[16] = {0};
  size_t cur_byte = 0;
  int double_colon_byte = -1;

  if (src.starts_with("::")) {
    double_colon_byte = 0;
    src.remove_prefix(2);
    if (src.empty()) {
      inline_memcpy(&dst.s6_addr, bytes, 16);
      return true;
    }
  } else if (src[0] == ':') {
    return false;
  }

  uint32_t val = 0;
  size_t num_digits = 0;
  size_t token_start = 0;

  for (size_t i = 0; i < src.size(); ++i) {
    char c = src[i];
    if (is_hex_char(c)) {
      if (++num_digits > 4)
        return false;
      if (num_digits == 1)
        token_start = i;
      val = (val << 4) | static_cast<uint32_t>(internal::b36_char_to_int(c));
    } else if (c == ':') {
      if (num_digits == 0) {
        if (double_colon_byte != -1)
          return false;
        double_colon_byte = static_cast<int>(cur_byte);
        continue;
      }
      if (i + 1 == src.size())
        return false; // Trailing single colon

      if (cur_byte + 2 > 16)
        return false;

      bytes[cur_byte++] = static_cast<uint8_t>(val >> 8);
      bytes[cur_byte++] = static_cast<uint8_t>(val & 0xff);
      val = 0;
      num_digits = 0;
    } else if (c == '.') {
      if (num_digits == 0 || cur_byte + 4 > 16)
        return false;

      cpp::string_view ipv4_str = src.substr(token_start);
      struct in_addr in4;
      if (!str_to_ipv4(ipv4_str, in4))
        return false;

      inline_memcpy(&bytes[cur_byte], &in4.s_addr, 4);
      cur_byte += 4;
      num_digits = 0;
      break;
    } else {
      return false;
    }
  }

  if (num_digits > 0) {
    if (cur_byte + 2 > 16)
      return false;
    bytes[cur_byte++] = static_cast<uint8_t>(val >> 8);
    bytes[cur_byte++] = static_cast<uint8_t>(val & 0xff);
  }

  if (double_colon_byte != -1) {
    if (cur_byte >= 16)
      return false;

    size_t bytes_after = cur_byte - static_cast<size_t>(double_colon_byte);
    for (size_t k = bytes_after; k > 0; --k)
      bytes[16 - bytes_after + (k - 1)] =
          bytes[static_cast<size_t>(double_colon_byte) + (k - 1)];

    size_t gap = 16 - cur_byte;
    for (size_t k = 0; k < gap; ++k)
      bytes[static_cast<size_t>(double_colon_byte) + k] = 0;
  } else if (cur_byte != 16) {
    return false;
  }

  inline_memcpy(&dst.s6_addr, bytes, 16);
  return true;
}

cpp::optional<in_addr_t> inet_addr(cpp::string_view src) {
  constexpr int IPV4_MAX_DOT_NUM = 3;
  in_addr_t parts[IPV4_MAX_DOT_NUM + 1] = {0};
  int dot_num = 0;

  for (; dot_num <= IPV4_MAX_DOT_NUM; ++dot_num) {
    // strtointeger skips leading whitespace and signs, and accepts C23 binary
    // string prefix (0b1.+2.-3. 4), but we don't want any of that, so we
    // explicitly check reject these constructs.
    if (src.empty() || !internal::isdigit(src[0]) || src.starts_with("0b") ||
        src.starts_with("0B"))
      return cpp::nullopt;

    auto result = internal::strtointeger<in_addr_t>(src.data(), 0, src.size());
    parts[dot_num] = result;

    if (result.has_error() || result.parsed_len == 0)
      return cpp::nullopt;
    src.remove_prefix(result.parsed_len);
    if (src.empty() || internal::isspace(src[0]))
      break;
    if (src[0] != '.')
      return cpp::nullopt;
    src.remove_prefix(1);
  }

  if (dot_num > IPV4_MAX_DOT_NUM)
    return cpp::nullopt;

  // converts the Internet host address src from the IPv4 numbers-and-dots
  // notation (a[.b[.c[.d]]]) into binary form (in network byte order)
  in_addr_t result = 0;
  for (int i = 0; i <= dot_num; ++i) {
    in_addr_t max_part = i == dot_num ? (0xffffffffu >> (8 * dot_num)) : 0xffu;
    if (parts[i] > max_part)
      return cpp::nullopt;
    int shift = i == dot_num ? 0 : 8 * (IPV4_MAX_DOT_NUM - i);
    result |= parts[i] << shift;
  }

  return Endian::to_big_endian(result);
}

namespace {

size_t ipv4_num_bytes(cpp::span<const uint8_t> src) {
  size_t result = 8; // four digits, three dots and '\0'
  for (uint8_t val : src)
    result += (val >= 10) + (val >= 100);
  return result;
}

size_t ipv4_to_str_unchecked(cpp::span<const uint8_t> src,
                             cpp::span<char> dst) {
  size_t pos = 0;
  for (unsigned i = 0; i < 4; ++i) {
    uint8_t val = src[i];
    if (val >= 100) {
      uint8_t cent = val / 100;
      uint8_t rem = val % 100;
      dst[pos++] = internal::int_to_b36_char(cent);
      dst[pos++] = internal::int_to_b36_char(rem / 10);
      dst[pos++] = internal::int_to_b36_char(rem % 10);
    } else if (val >= 10) {
      dst[pos++] = internal::int_to_b36_char(val / 10);
      dst[pos++] = internal::int_to_b36_char(val % 10);
    } else {
      dst[pos++] = internal::int_to_b36_char(val);
    }
    dst[pos++] = i < 3 ? '.' : '\0';
  }
  return pos;
}

size_t ipv6_to_str_unchecked(const struct in6_addr &src, cpp::span<char> dst) {
  // Find the longest run of zeroes to compress to "::"
  struct Run {
    unsigned start = 0;
    unsigned len = 0;
  };
  Run best, current;
  for (unsigned i = 0; i < 8; ++i) {
    uint16_t val = src.s6_addr16[i];
    if (val == 0) {
      ++current.len;
    } else {
      // In case of ties, the first sequence wins.
      if (current.len > best.len)
        best = current;
      current = {i + 1, 0};
    }
  }
  if (current.len > best.len)
    best = current;

  bool is_mapped =
      best.start == 0 &&
      (best.len == 6 || (best.len == 5 && src.s6_addr16[5] == 0xffff));
  unsigned num_words = is_mapped ? 6 : 8;

  size_t pos = 0;
  auto append_word = [&](unsigned i) {
    uint16_t word = Endian::from_big_endian(src.s6_addr16[i]);
    // This isn't using int_to_b36_char because it's large intermediate
    // representation prevents append_word from being inlined.
    static constexpr char DIGITS[] = "0123456789abcdef";
    if (word >= 0x1000) {
      dst[pos] = DIGITS[word >> 12];
      dst[pos + 1] = DIGITS[(word >> 8) & 0xf];
      dst[pos + 2] = DIGITS[(word >> 4) & 0xf];
      dst[pos + 3] = DIGITS[word & 0xf];
      pos += 4;
    } else if (word >= 0x100) {
      dst[pos] = DIGITS[word >> 8];
      dst[pos + 1] = DIGITS[(word >> 4) & 0xf];
      dst[pos + 2] = DIGITS[word & 0xf];
      pos += 3;
    } else if (word >= 0x10) {
      dst[pos] = DIGITS[(word >> 4) & 0xf];
      dst[pos + 1] = DIGITS[word & 0xf];
      pos += 2;
    } else {
      dst[pos] = DIGITS[word];
      pos += 1;
    }
  };

  if (best.len < 2) {
    // No compression
    for (unsigned i = 0; i < 7; ++i) {
      append_word(i);
      dst[pos++] = ':';
    }
    append_word(7);
    dst[pos++] = '\0';
    return pos;
  }

  // Left part
  for (unsigned i = 0; i < best.start; ++i) {
    append_word(i);
    dst[pos++] = ':';
  }
  // Compressed part
  if (best.start == 0)
    dst[pos++] = ':';
  dst[pos++] = ':';

  // Right part (if it exists)
  if (best.start + best.len < num_words) {
    unsigned end = num_words - 1;
    for (unsigned i = best.start + best.len; i < end; ++i) {
      append_word(i);
      dst[pos++] = ':';
    }
    append_word(end);
    if (num_words == 6)
      dst[pos++] = ':';
  }

  if (is_mapped) {
    cpp::span<const uint8_t> ipv4_part(src.s6_addr + 12, 4);
    pos += ipv4_to_str_unchecked(ipv4_part, dst.subspan(pos));
  } else {
    dst[pos++] = '\0';
  }

  return pos;
}

} // anonymous namespace

bool ipv4_to_str(const struct in_addr &src, cpp::span<char> dst) {
  cpp::span<const uint8_t> addr(reinterpret_cast<const uint8_t *>(&src), 4);

  if (dst.size() < INET_ADDRSTRLEN) {
    if (dst.size() < ipv4_num_bytes(addr))
      return false;
  }

  ipv4_to_str_unchecked(addr, dst);
  return true;
}

bool ipv6_to_str(const struct in6_addr &src, cpp::span<char> dst) {
  if (dst.size() >= INET6_ADDRSTRLEN) {
    ipv6_to_str_unchecked(src, dst);
    return true;
  }
  char buf[INET6_ADDRSTRLEN];
  size_t len = ipv6_to_str_unchecked(src, buf);
  LIBC_ASSERT(len < INET6_ADDRSTRLEN);
  if (len > dst.size())
    return false;
  inline_memcpy(dst.data(), buf, len);
  return true;
}

} // namespace net
} // namespace LIBC_NAMESPACE_DECL

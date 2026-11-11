//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Internal utilities for Ethernet address manipulation.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_NETINET_ETHER_ETHER_UTILS_H
#define LLVM_LIBC_SRC_NETINET_ETHER_ETHER_UTILS_H

#include "hdr/stdint_proxy.h"
#include "hdr/types/size_t.h"
#include "hdr/types/struct_ether_addr.h"
#include "src/__support/CPP/optional.h"
#include "src/__support/CPP/span.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/common.h"
#include "src/__support/ctype_utils.h"
#include "src/__support/macros/config.h"
#include "src/__support/net/hex.h"

namespace LIBC_NAMESPACE_DECL {
namespace net {

/// Maximum length of an Ethernet address string "xx:xx:xx:xx:xx:xx\0".
inline constexpr size_t ETHER_ADDR_STR_LEN = 18;

/// Formats \p addr into \p buf (which must be at least ETHER_ADDR_STR_LEN
/// bytes) as a colon-separated lowercase hexadecimal string and returns
/// `buf.data()`.
LIBC_INLINE char *ether_ntoa_r(const struct ether_addr &addr,
                               cpp::span<char> buf) {
  size_t pos = 0;
  for (size_t i = 0; i < 6; ++i) {
    uint8_t val = addr.ether_addr_octet[i];
    if (val >= 0x10)
      buf[pos++] = internal::int_to_b36_char(val >> 4);
    buf[pos++] = internal::int_to_b36_char(val & 0xf);
    buf[pos++] = i < 5 ? ':' : '\0';
  }
  return buf.data();
}

/// Parses \p str as an Ethernet address in the colon-separated hexadecimal
/// notation ("xx:xx:xx:xx:xx:xx"), where each octet consists of one or two
/// hexadecimal digits. Like musl (but unlike glibc), trailing characters are
/// not allowed. Returns nullopt if the string is not a valid address.
LIBC_INLINE cpp::optional<struct ether_addr> ether_aton(cpp::string_view str) {
  struct ether_addr addr;
  for (size_t i = 0; i < 6; ++i) {
    if (i > 0) {
      if (!str.starts_with(':'))
        return cpp::nullopt;
      str.remove_prefix(1);
    }
    cpp::optional<uint8_t> octet = consume_hex<uint8_t>(str);
    if (!octet)
      return cpp::nullopt;
    addr.ether_addr_octet[i] = *octet;
  }
  if (!str.empty())
    return cpp::nullopt;
  return addr;
}

} // namespace net
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_NETINET_ETHER_ETHER_UTILS_H

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <__config>
#include <__functional/hash.h>
#include <__stacktrace/basic_stacktrace.h>
#include <__stacktrace/stacktrace_entry.h>
#include <string>

#if _LIBCPP_HAS_LOCALIZATION
#  include <iostream>
#  include <sstream>
#endif //_LIBCPP_HAS_LOCALIZATION

#include "stacktrace/images.h"

_LIBCPP_BEGIN_NAMESPACE_STD
_LIBCPP_BEGIN_EXPLICIT_ABI_ANNOTATIONS

namespace __stacktrace {

#if _LIBCPP_HAS_LOCALIZATION

ostream& _Entry::__write_to(ostream& __os) const {
  // printf-style format to a small buffer, to avoid messing with stream (with `setw` etc.)
  char ubuf[25]{};
  if constexpr (sizeof(void*) > 4) {
    // Although 64-bit addresses are 16 nibbles long, they're often <= 0x7fff_ffff_ffff
    snprintf(ubuf, sizeof(ubuf) - 1, "0x%012llx", (unsigned long long)(__addr_));
  } else {
    snprintf(ubuf, sizeof(ubuf) - 1, "0x%08lx", (unsigned long)(__addr_));
  }
  __os << ubuf;

  if (__desc_.__view().size()) {
    __os << ": " << __desc_.__view();
  }

  if (__file_.__view().size()) {
    __os << ": " << __file_.__view();
  }

  if (__line_) {
    snprintf(ubuf, sizeof(ubuf) - 1, "%u", __line_);
    __os << ":" << ubuf;
  }

  return __os;
}

string _Entry::__to_string() const {
  stringstream __ss;
  __write_to(__ss);
  return __ss.str();
}

#endif // _LIBCPP_HAS_LOCALIZATION

uintptr_t _Entry::__adjusted_addr() const {
  auto sub = __image_ ? __image_->slide_ : 0;
  return __addr_ - sub;
}

size_t _Entry::__hash_code() const { return std::__hash_memory(&__addr_, sizeof(uintptr_t)); }

} // namespace __stacktrace

_LIBCPP_END_EXPLICIT_ABI_ANNOTATIONS
_LIBCPP_END_NAMESPACE_STD

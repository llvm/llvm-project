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
#  include <iomanip>
#  include <iostream>
#  include <sstream>
#endif //_LIBCPP_HAS_LOCALIZATION

#include "stacktrace/images.h"

_LIBCPP_BEGIN_NAMESPACE_STD

namespace __stacktrace {

#if _LIBCPP_HAS_LOCALIZATION

_LIBCPP_EXPORTED_FROM_ABI ostream& _Entry::__write_to(ostream& __os) const {
  // Although 64-bit addresses are 16 nibbles long, they're often <= 0x7fff_ffff_ffff
  constexpr static int __k_addr_width = (sizeof(void*) > 4) ? 12 : 8;

  // TODO: should not use os with `hex` etc.

  __os << "0x" << std::hex << std::setfill('0') << std::setw(__k_addr_width) << __addr_;
  if (__desc_.__view().size()) {
    __os << ": " << __desc_.__view();
  }
  if (__file_.__view().size()) {
    __os << ": " << __file_.__view();
  }
  if (__line_) {
    __os << ":" << std::dec << __line_;
  }
  return __os;
}

_LIBCPP_EXPORTED_FROM_ABI string _Entry::__to_string() const {
  stringstream __ss;
  __write_to(__ss);
  return __ss.str();
}

#endif // _LIBCPP_HAS_LOCALIZATION

uintptr_t _Entry::__adjusted_addr() const {
  auto sub = __image_ ? __image_->slide_ : 0;
  return __addr_ - sub;
}

_LIBCPP_EXPORTED_FROM_ABI size_t _Entry::__hash_code() const { return std::__hash_memory(&__addr_, sizeof(uintptr_t)); }

} // namespace __stacktrace

_LIBCPP_END_NAMESPACE_STD

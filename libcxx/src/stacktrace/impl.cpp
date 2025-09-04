//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <__config>
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

_LIBCPP_EXPORTED_FROM_ABI ostream& entry_base::write_to(ostream& __os) const {
  // Although 64-bit addresses are 16 nibbles long, they're often <= 0x7fff_ffff_ffff
  constexpr static int __k_addr_width = (sizeof(void*) > 4) ? 12 : 8;

  __os << "0x" << std::hex << std::setfill('0') << std::setw(__k_addr_width) << __addr_;
  if (__desc_) {
    __os << ": " << __desc_->view();
  }
  if (__file_) {
    __os << ": " << __file_->view();
  }
  if (__line_) {
    __os << ":" << std::dec << __line_;
  }
  return __os;
}

_LIBCPP_EXPORTED_FROM_ABI ostream& base::write_to(std::ostream& __os) const {
  auto iters = __entry_iters_();
  auto count = iters.size();
  if (!count) {
    __os << "(empty stacktrace)";
  } else {
    for (size_t __i = 0; __i < count; __i++) {
      // Insert newlines between entries (but not before the first or after the last)
      if (__i) {
        __os << '\n';
      }
      __os << "  frame " << std::setw(3) << std::setfill(' ') << std::dec << (__i + 1) << ": "
           << *(stacktrace_entry const*)(iters.data() + __i);
    }
  }
  return __os;
}

_LIBCPP_EXPORTED_FROM_ABI string entry_base::to_string() const {
  stringstream __ss;
  write_to(__ss);
  return __ss.str();
}

_LIBCPP_EXPORTED_FROM_ABI string base::to_string() const {
  stringstream __ss;
  write_to(__ss);
  return __ss.str();
}

#endif // _LIBCPP_HAS_LOCALIZATION

_LIBCPP_HIDE_FROM_ABI uintptr_t entry_base::adjusted_addr() const {
  auto sub = __image_ ? __image_->slide_ : 0;
  return __addr_ - sub;
}

} // namespace __stacktrace

_LIBCPP_END_NAMESPACE_STD

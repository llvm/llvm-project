//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <__config>
#include <__stacktrace/basic_stacktrace.h>
#include <__stacktrace/images.h>
#include <iomanip>
#include <iostream>
#include <sstream>

_LIBCPP_BEGIN_NAMESPACE_STD

namespace __stacktrace {

ostream& entry_base::write_to(ostream& __os) const {
  // Although 64-bit addresses are 16 nibbles long, they're often <= 0x7fff_ffff_ffff
  constexpr static int __k_addr_width = (sizeof(void*) > 4) ? 12 : 8;

  __os << "0x" << std::hex << std::setfill('0') << std::setw(__k_addr_width) << __addr_;
  if (__desc_.size()) {
    __os << ": " << __desc_;
  }
  if (__file_.size()) {
    __os << ": " << __file_;
  }
  if (__line_) {
    __os << ":" << std::dec << __line_;
  }
  return __os;
}

ostream& base::write_to(std::ostream& __os) const {
  auto __count = __entries_size_();
  if (!__count) {
    __os << "(empty stacktrace)";
  } else {
    for (size_t __i = 0; __i < __count; __i++) {
      // Insert newlines between entries (but not before the first or after the last)
      if (__i) {
        __os << '\n';
      }
      __os << "  frame " << std::setw(3) << std::setfill(' ') << std::dec << (__i + 1) << ": "
           << (stacktrace_entry&)__entry_at_(__i);
    }
  }
  return __os;
}

string entry_base::to_string() const {
  stringstream __ss;
  write_to(__ss);
  return __ss.str();
}

string base::to_string() const {
  stringstream __ss;
  write_to(__ss);
  return __ss.str();
}

uintptr_t entry_base::adjusted_addr() const {
  auto sub = __image_ ? __image_->slide_ : 0;
  return __addr_ - sub;
}

} // namespace __stacktrace

_LIBCPP_END_NAMESPACE_STD

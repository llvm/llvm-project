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

_LIBCPP_BEGIN_NAMESPACE_STD

namespace __stacktrace {

#if _LIBCPP_HAS_LOCALIZATION

_LIBCPP_EXPORTED_FROM_ABI ostream& _Trace::write_to(std::ostream& __os) const {
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

_LIBCPP_EXPORTED_FROM_ABI string _Trace::to_string() const {
  stringstream __ss;
  write_to(__ss);
  return __ss.str();
}

#endif // _LIBCPP_HAS_LOCALIZATION

} // namespace __stacktrace

_LIBCPP_END_NAMESPACE_STD

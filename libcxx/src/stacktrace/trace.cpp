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
#include <stacktrace>
#include <string>

#if _LIBCPP_HAS_LOCALIZATION
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

      stacktrace_entry& entry = *reinterpret_cast<stacktrace_entry*>(iters.data() + __i);

      // printf-style format to a small buffer, to avoid messing with stream (with `setw` etc.)
      char index_str[21];
      snprintf(index_str, sizeof(index_str), "%3zu", __i + 1);
      __os << "  frame " << index_str << ": " << entry;
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

_LIBCPP_EXPORTED_FROM_ABI size_t _Trace::hash() const {
  size_t __ret = size_t(0xc3a5c85c97cb3127ull); // taken from __functional/hash.h
  for (_Entry const& __e : __entry_iters_()) {
    __ret = (__ret << 1) ^ __e.hash();
  }
  return __ret;
}

} // namespace __stacktrace

_LIBCPP_END_NAMESPACE_STD

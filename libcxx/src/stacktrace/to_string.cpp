//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <__config>
#include <__config_site>

#include <iomanip>
#include <ios>
#include <iostream>
#include <sstream>
#include <string>

#include <__stacktrace/basic.h>
#include <__stacktrace/entry.h>
#include <__stacktrace/to_string.h>

_LIBCPP_BEGIN_NAMESPACE_STD

// `to_string`-related non-member functions

_LIBCPP_EXPORTED_FROM_ABI string to_string(const stacktrace_entry& __entry) {
  return __stacktrace::__to_string()(__entry);
}

_LIBCPP_EXPORTED_FROM_ABI ostream& operator<<(ostream& __os, const stacktrace_entry& __entry) {
  __stacktrace::__to_string()(__os, __entry);
  return __os;
}

namespace __stacktrace {

/*
 * `to_string` Helpers
 */

_LIBCPP_EXPORTED_FROM_ABI void __to_string::operator()(ostream& __os, std::stacktrace_entry const& entry) {
  // Although 64-bit addresses are 16 nibbles long, they're often <= 0x7fff_ffff_ffff
  constexpr static int __k_addr_width = (sizeof(void*) > 4) ? 12 : 8;

  __os << "0x" << std::hex << std::setfill('0') << std::setw(__k_addr_width) << entry.native_handle();
  if (!entry.description().empty()) {
    __os << ": " << entry.description();
  }
  if (!entry.source_file().empty()) {
    __os << ": " << entry.source_file();
  }
  if (entry.source_line()) {
    __os << ":" << std::dec << entry.source_line();
  }
}

_LIBCPP_EXPORTED_FROM_ABI void
__to_string::operator()(ostream& __os, std::stacktrace_entry const* __entries, size_t __count) {
  /*
   * Print each entry as a line, as per `operator()`, with additional whitespace
   * at the start of the line, and only a newline added at the end:
   *
   *   frame   1: 0xbeefbeefbeef: _symbol_name: /path/to/file.cc:123
   */
  if (!__count) {
    __os << "(empty stacktrace)";
  } else {
    for (size_t __i = 0; __i < __count; __i++) {
      if (__i) {
        // Insert newlines between entries (but not before the first or after the last)
        __os << std::endl;
      }
      __os << "  frame " << std::setw(3) << std::setfill(' ') << std::dec << (__i + 1) << ": ";
      (*this)(__os, __entries[__i]);
    }
  }
}

_LIBCPP_EXPORTED_FROM_ABI string __to_string::operator()(std::stacktrace_entry const& entry) {
  stringstream __ss;
  (*this)(__ss, entry);
  return __ss.str();
}

_LIBCPP_EXPORTED_FROM_ABI string __to_string::operator()(std::stacktrace_entry const* __entries, size_t __count) {
  stringstream __ss;
  (*this)(__ss, __entries, __count);
  return __ss.str();
}

} // namespace __stacktrace

_LIBCPP_END_NAMESPACE_STD

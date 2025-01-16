//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_STACKTRACE_UTILS_DEBUG
#define _LIBCPP_STACKTRACE_UTILS_DEBUG

#include <__config>
#include <iostream>

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __stacktrace {

/** Debug-message output stream.  If `LIBCXX_STACKTRACE_DEBUG` is defined in the environment
or as a macro with exactly the string `1` then this is enabled (prints to `std::cerr`);
otherwise its does nothing by returning a dummy stream. */
struct _LIBCPP_HIDE_FROM_ABI debug : std::ostream {
  _LIBCPP_HIDE_FROM_ABI virtual ~debug() = default;

  _LIBCPP_HIDE_FROM_ABI static bool enabled() {
#if defined(LIBCXX_STACKTRACE_DEBUG) && LIBCXX_STACKTRACE_DEBUG == 1
    return true;
#else
    static bool ret = [] {
      auto const* val = getenv("LIBCXX_STACKTRACE_DEBUG");
      return val && !strncmp(val, "1", 1);
    }();
    return ret;
#endif
  }

  /** No-op output stream. */
  struct _LIBCPP_HIDE_FROM_ABI dummy_ostream final : std::ostream {
    _LIBCPP_HIDE_FROM_ABI virtual ~dummy_ostream() = default;
    friend std::ostream& operator<<(dummy_ostream& bogus, auto const&) { return bogus; }
  };

  friend std::ostream& operator<<(debug& dp, auto const& val) {
    static dummy_ostream kdummy;
    if (!enabled()) {
      return kdummy;
    }
    std::cerr << val;
    return std::cerr;
  }
};

} // namespace __stacktrace
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STACKTRACE_UTILS_DEBUG

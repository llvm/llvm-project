//===----------------------------------------------------------------------===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===////

#ifndef FILESYSTEM_FORMAT_STRING_H
#define FILESYSTEM_FORMAT_STRING_H

#include <__assert>
#include <__config>
#include <__utility/scope_guard.h>
#include <array>
#include <cstdarg>
#include <cstddef>
#include <cstdio>
#include <string>

#if defined(_LIBCPP_WIN32API)
#  define PATHSTR(x) (L##x)
#  define PATH_CSTR_FMT "\"%ls\""
#else
#  define PATHSTR(x) (x)
#  define PATH_CSTR_FMT "\"%s\""
#endif

_LIBCPP_BEGIN_NAMESPACE_FILESYSTEM

namespace detail {

inline _LIBCPP_ATTRIBUTE_FORMAT(__printf__, 1, 0) string vformat_string(const char* msg, va_list ap) {
  array<char, 256> buf;

  va_list apcopy;
  va_copy(apcopy, ap);
  int size = ::vsnprintf(buf.data(), buf.size(), msg, apcopy);
  va_end(apcopy);

  string result;
  if (static_cast<size_t>(size) < buf.size()) {
    result.assign(buf.data(), static_cast<size_t>(size));
  } else {
    // we did not provide a long enough buffer on our first attempt. The
    // return value is the number of bytes (excluding the null byte) that are
    // needed for formatting.
    result.resize_and_overwrite(size, [&](char* res, size_t n) { return ::vsnprintf(res, n, msg, ap); });
    _LIBCPP_ASSERT_INTERNAL(static_cast<size_t>(size) == result.size(),
                            "vsnprintf did not result in the same number of characters as the first attempt?");
  }
  return result;
}

inline _LIBCPP_ATTRIBUTE_FORMAT(__printf__, 1, 2) string format_string(const char* msg, ...) {
  string ret;
  va_list ap;
  va_start(ap, msg);
  __scope_guard guard([&] { va_end(ap); });
  ret = detail::vformat_string(msg, ap);
  return ret;
}

} // namespace detail

_LIBCPP_END_NAMESPACE_FILESYSTEM

#endif // FILESYSTEM_FORMAT_STRING_H

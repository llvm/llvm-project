//===-----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___SUPPORT_IBMVASPRINTF_H
#define _LIBCPP___SUPPORT_IBMVASPRINTF_H

#include <cstdarg> // va_copy, va_end
#include <cstdio>  // vsnprintf
#include <cstdlib> // malloc, realloc

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __ibm {

inline _LIBCPP_HIDE_FROM_ABI
_LIBCPP_ATTRIBUTE_FORMAT(__printf__, 2, 0) int __vasprintf(char** __strp, const char* __fmt, va_list __ap) {
  const size_t buff_size = 256;
  if ((*__strp = (char*)std::malloc(buff_size)) == nullptr) {
    return -1;
  }

  va_list ap_copy;
  // va_copy may not be provided by the C library in C++03 mode.
#if defined(_LIBCPP_CXX03_LANG) && __has_builtin(__builtin_va_copy)
#  if defined(__MVS__) && !defined(_VARARG_EXT_)
  __builtin_zos_va_copy(ap_copy, __ap);
#  else
  __builtin_va_copy(ap_copy, __ap);
#  endif
#else
  va_copy(ap_copy, __ap);
#endif
  int str_size = vsnprintf(*__strp, buff_size, __fmt, ap_copy);
  va_end(ap_copy);

  if ((size_t)str_size >= buff_size) {
    if ((*__strp = (char*)std::realloc(*__strp, str_size + 1)) == nullptr) {
      return -1;
    }
    str_size = vsnprintf(*__strp, str_size + 1, __fmt, __ap);
  }
  return str_size;
}

} // namespace __ibm
_LIBCPP_END_NAMESPACE_STD
#endif // _LIBCPP___SUPPORT_IBMVASPRINTF_H

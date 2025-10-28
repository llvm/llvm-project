//===-----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___SUPPORT_IBMVASPRINTF_H
#define _LIBCPP___SUPPORT_IBMVASPRINTF_H

#include <cstdlib>  // malloc, realloc
#include <stdarg.h> // va_copy, va_end
#include <stdio.h>  // vsnprintf

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __ibm {

inline _LIBCPP_HIDE_FROM_ABI
_LIBCPP_ATTRIBUTE_FORMAT(__printf__, 2, 0) int vasprintf(char** strp, const char* fmt, va_list ap) {
  const size_t buff_size = 256;
  if ((*strp = (char*)malloc(buff_size)) == nullptr) {
    return -1;
  }

  va_list ap_copy;
  // va_copy may not be provided by the C library in C++03 mode.
#if defined(_LIBCPP_CXX03_LANG) && __has_builtin(__builtin_va_copy)
#  if defined(__MVS__) && !defined(_VARARG_EXT_)
  __builtin_zos_va_copy(ap_copy, ap);
#  else
  __builtin_va_copy(ap_copy, ap);
#  endif
#else
  va_copy(ap_copy, ap);
#endif
  int str_size = vsnprintf(*strp, buff_size, fmt, ap_copy);
  va_end(ap_copy);

  if ((size_t)str_size >= buff_size) {
    if ((*strp = (char*)realloc(*strp, str_size + 1)) == nullptr) {
      return -1;
    }
    str_size = vsnprintf(*strp, str_size + 1, fmt, ap);
  }
  return str_size;
}

} // namespace __ibm
_LIBCPP_END_NAMESPACE_STD
#endif // _LIBCPP___SUPPORT_IBMVASPRINTF_H

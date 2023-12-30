//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <__config>
#include <cassert>
#include <cstdio>
#include <fstream>

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 26
template <class _CharT, class _Traits>
basic_filebuf<_CharT, _Traits>::native_handle_type basic_filebuf<_CharT, _Traits>::native_handle() {
  assert(is_open());
  // __file_ is a FILE*
#  if defined(_LIBCPP_WIN32API)
  // https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/get-osfhandle?view=msvc-170
  intptr_t __handle = _get_osfhandle(::fileno(__file_));
  if (__handle == -1)
    return nullptr;
  return reinterpret_cast<void*>(__handle);
#  else
  return ::fileno(__file_);
#  endif
}
#endif

_LIBCPP_END_NAMESPACE_STD

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <__config>
#include <cstdio>
#include <fstream>

#if defined(_LIBCPP_WIN32API)
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <io.h>
#  include <windows.h>
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 26
_LIBCPP_EXPORTED_FROM_ABI __filebuf_native_handle_type __filebuf_native_handle(FILE* __file) noexcept {
#  if defined(_LIBCPP_WIN32API)
  // https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/get-osfhandle?view=msvc-170
  intptr_t __handle = _get_osfhandle(::fileno(__file));
  if (__handle == -1)
    return nullptr;
  return reinterpret_cast<void*>(__handle);
#  else
  return ::fileno(__file);
#  endif
}
#endif

_LIBCPP_END_NAMESPACE_STD

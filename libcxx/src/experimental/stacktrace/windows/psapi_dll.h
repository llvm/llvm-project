//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_STACKTRACE_WINDOWS_PSAPI_DLL_H
#define _LIBCPP_STACKTRACE_WINDOWS_PSAPI_DLL_H

#include "../common/config.h"

#if defined(_LIBCPP_STACKTRACE_WINDOWS)
// windows.h must be first
#  include <windows.h>
// other windows-specific headers
#  include <dbghelp.h>
#  define PSAPI_VERSION 1
#  include <psapi.h>
// standard headers
#  include <cstdlib>
#  include <mutex>

#  include <experimental/__stacktrace/detail/entry.h>

#  include "../common/debug.h"
#  include "dll.h"

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __stacktrace {

// clang-format off

struct psapi_dll final : dll {
  virtual ~psapi_dll() = default;
  static psapi_dll& get() { static psapi_dll ret; return ret; }

  bool  (*EnumProcessModules)   (HANDLE, HMODULE*, DWORD, DWORD*);
  bool  (*GetModuleInformation) (HANDLE, HMODULE, MODULEINFO*, DWORD);
  DWORD (*GetModuleBaseName)    (HANDLE, HMODULE, char**, DWORD);

  psapi_dll() : dll("psapi.dll") {
    if (!get_func(&EnumProcessModules, "EnumProcessModules")) { return; }
    if (!get_func(&GetModuleInformation, "GetModuleInformation")) { return; }
    if (!get_func(&GetModuleBaseName, "GetModuleBaseNameA")) { return; }
    valid_ = true;
  }
};

} // namespace __stacktrace
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STACKTRACE_WINDOWS
#endif // _LIBCPP_STACKTRACE_WINDOWS_PSAPI_DLL_H

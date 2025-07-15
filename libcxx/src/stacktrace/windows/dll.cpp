//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <__config>

#if defined(_LIBCPP_WIN32API)

#  include <__stacktrace/base.h>

#  include "stacktrace/windows/dll.h"
#  include "stacktrace/windows/impl.h"

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __stacktrace {

namespace {

// Initialized once, in first `win_impl` construction.
// Protected by mutex within the `win_impl` constructor.
HANDLE proc;
HMODULE exe;
IMAGE_NT_HEADERS* ntHeaders;
bool globalInitialized{false};

// Globals used across invocations of the functions below.
// Protected by mutex within the `win_impl` constructor.
bool symsInitialized{false};
HMODULE moduleHandles[1024];
size_t moduleCount; // 0 IFF module enumeration failed

} // namespace

dll::~dll() { FreeLibrary(module_); }

dll::dll(char const* name) : name_(name), module_(LoadLibrary(name)) {}

dbghelp_dll::~dbghelp_dll() = default;

dbghelp_dll& dbghelp_dll::get() {
  static dbghelp_dll ret;
  return ret;
}

dbghelp_dll::dbghelp_dll() : dll("dbghelp.dll") {
  // clang-format off
if (!get_func(&ImageNtHeader, "ImageNtHeader")) { return; }
  if (!get_func(&StackWalk64, "StackWalk64")) { return; }
  if (!get_func(&SymCleanup, "SymCleanup")) { return; }
  if (!get_func(&SymFunctionTableAccess64, "SymFunctionTableAccess64")) { return; }
  if (!get_func(&SymGetLineFromAddr64, "SymGetLineFromAddr64")) { return; }
  if (!get_func(&SymGetModuleBase64, "SymGetModuleBase64")) { return; }
  if (!get_func(&SymGetOptions, "SymGetOptions")) { return; }
  if (!get_func(&SymGetSymFromAddr64, "SymGetSymFromAddr64")) { return; }
  if (!get_func(&SymInitialize, "SymInitialize")) { return; }
  if (!get_func(&SymLoadModule64, "SymLoadModule64")) { return; }
  if (!get_func(&SymSetOptions, "SymSetOptions")) { return; }
  valid_ = true;
  // clang-format on
}

psapi_dll::~psapi_dll() = default;

psapi_dll& psapi_dll::get() {
  static psapi_dll ret;
  return ret;
}

psapi_dll::psapi_dll() : dll("psapi.dll") {
  // clang-format off
if (!get_func(&EnumProcessModules, "EnumProcessModules")) { return; }
  if (!get_func(&GetModuleInformation, "GetModuleInformation")) { return; }
  if (!get_func(&GetModuleBaseName, "GetModuleBaseNameA")) { return; }
  valid_ = true;
  // clang-format on
}

} // namespace __stacktrace
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_WIN32API

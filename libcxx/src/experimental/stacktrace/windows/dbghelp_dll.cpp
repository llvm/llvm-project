//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

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

#  include "dll.h"
#  include <experimental/__stacktrace/detail/entry.h>

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __stacktrace {

dbghelp_dll::~dbghelp_dll() = default;

dbghelp_dll& dbghelp_dll::get() {
  dbghelp_dll ret;
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

} // namespace __stacktrace
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STACKTRACE_USE_DBGHELP

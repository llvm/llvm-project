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
#  include <__stacktrace/entry.h>

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __stacktrace {

psapi_dll::~psapi_dll() = default;

psapi_dll& psapi_dll::get() {
  psapi_dll ret;
  return ret;
}

psapi_dll() : dll("psapi.dll") {
  // clang-format off
if (!getFunc(&EnumProcessModules, "EnumProcessModules")) { return; }
  if (!getFunc(&GetModuleInformation, "GetModuleInformation")) { return; }
  if (!getFunc(&GetModuleBaseName, "GetModuleBaseNameA")) { return; }
  valid_ = true;
  // clang-format on
}

} // namespace __stacktrace
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STACKTRACE_USE_DBGHELP

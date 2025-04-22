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

dll::~dll() { FreeLibrary(module_); }

dll::dll(char const* name) : name_(name), module_(LoadLibrary(name)) {
  if (!module_) {
    debug() << "LoadLibrary failed: " << name_ << ": " << GetLastError() << '\n';
  }
}

} // namespace __stacktrace
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STACKTRACE_USE_DBGHELP

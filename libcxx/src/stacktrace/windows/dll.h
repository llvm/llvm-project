//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_STACKTRACE_WINDOWS_DLL_H
#define _LIBCPP_STACKTRACE_WINDOWS_DLL_H

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

#  include <__stacktrace/entry.h>

#  include "../common/debug.h"

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __stacktrace {

// clang-format off

struct dll {
  char const* name_;
  HMODULE module_;
  /** Set to true in subclass's ctor if initialized successfully. */
  bool valid_{false};

  operator bool() const { return valid_; }

  explicit dll(char const* name)
      : name_(name), module_(LoadLibrary(name)) {
    if (!module_) {
      debug() << "LoadLibrary failed: "
              << name_ << ": " << GetLastError() << '\n';
    }
  }

  virtual ~dll() { FreeLibrary(module_); }

  template <typename F>
  bool get_func(F* func, char const* name) {
    if (!(*func = (F)GetProcAddress(module_, name))) {
      debug() << "GetProcAddress failed: "
              << name << "' (" << name_ << "): "
              << GetLastError() << "\n";
      return false;
    }
    return true;
  }
};

} // namespace __stacktrace
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STACKTRACE_WINDOWS
#endif // _LIBCPP_STACKTRACE_WINDOWS_DLL_H

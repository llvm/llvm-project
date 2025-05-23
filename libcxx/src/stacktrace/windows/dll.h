//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_STACKTRACE_WIN_DLL
#define _LIBCPP_STACKTRACE_WIN_DLL

#include <__config>
#if defined(_LIBCPP_WIN32API)

// windows.h must be first
#  include <windows.h>
// other windows-specific headers
#  include <dbghelp.h>
#  define PSAPI_VERSION 1
#  include <psapi.h>

#  include <__config_site>
#  include <cstddef>
#  include <cstdlib>
#  include <mutex>
#  include <stacktrace>

#  include "stacktrace/utils/debug.h"

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __stacktrace {

#  if defined(_LIBCPP_WIN32API)

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

struct dbghelp_dll final : dll {
  virtual ~dbghelp_dll() = default;
  static dbghelp_dll& get() { static dbghelp_dll ret; return ret; }

  IMAGE_NT_HEADERS* (*ImageNtHeader)(void*);
  bool    (*StackWalk64)        (DWORD, HANDLE, HANDLE, STACKFRAME64*, void*, void*, void*, void*, void*);
  bool    (*SymCleanup)         (HANDLE);
  void*   (*SymFunctionTableAccess64)(HANDLE, DWORD64);
  bool    (*SymGetLineFromAddr64)(HANDLE, DWORD64, DWORD*, IMAGEHLP_LINE64*);
  DWORD64 (*SymGetModuleBase64) (HANDLE, DWORD64);
  DWORD   (*SymGetOptions)      ();
  bool    (*SymGetSymFromAddr64)(HANDLE, DWORD64, DWORD64*, IMAGEHLP_SYMBOL64*);
  bool    (*SymInitialize)      (HANDLE, char const*, bool);
  DWORD64 (*SymLoadModule64)    (HANDLE, HANDLE, char const*, char const*, void*, DWORD);
  DWORD   (*SymSetOptions)      (DWORD);

  dbghelp_dll() : dll("dbghelp.dll") {
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
  }
};

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

#endif

} // namespace __stacktrace
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_WIN32API
#endif // _LIBCPP_STACKTRACE_WIN_DLL

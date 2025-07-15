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

#  include <__stacktrace/base.h>
#  include <cstddef>
#  include <cstdlib>
#  include <mutex>

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __stacktrace {

#  if defined(_LIBCPP_WIN32API)

// clang-format off

struct dll {
  char const* name_;
  HMODULE module_;
  /** Set to true in subclass's ctor if initialized successfully. */
  bool valid_{false};

  virtual ~dll();
  explicit dll(char const* name);

  operator bool() const { return valid_; }

  template <typename F>
  bool get_func(F* func, char const* name) {
    *func = (F)GetProcAddress(module_, name);
    return func != nullptr;
  }
};

struct dbghelp_dll final : dll {
  virtual ~dbghelp_dll();
  dbghelp_dll();

  static dbghelp_dll& get();

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
};

struct psapi_dll final : dll {
  virtual ~psapi_dll();
  psapi_dll();

  static psapi_dll& get();

  bool  (*EnumProcessModules)   (HANDLE, HMODULE*, DWORD, DWORD*);
  bool  (*GetModuleInformation) (HANDLE, HMODULE, MODULEINFO*, DWORD);
  DWORD (*GetModuleBaseName)    (HANDLE, HMODULE, char**, DWORD);
};

#endif

} // namespace __stacktrace
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_WIN32API
#endif // _LIBCPP_STACKTRACE_WIN_DLL

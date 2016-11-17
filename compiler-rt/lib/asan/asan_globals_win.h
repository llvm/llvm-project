//===-- asan_globals_win.h --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Interface to the Windows-specific global management code. Separated into a
// standalone header to allow inclusion from asan_win_dynamic_runtime_thunk,
// which defines symbols that clash with other sanitizer headers.
//
//===----------------------------------------------------------------------===//

#ifndef ASAN_GLOBALS_WIN_H
#define ASAN_GLOBALS_WIN_H

#if !defined(_MSC_VER)
#error "this file is Windows-only, and uses MSVC pragmas"
#endif

#if defined(_WIN64)
#define SANITIZER_SYM_PREFIX
#else
#define SANITIZER_SYM_PREFIX "_"
#endif

// Use this macro to force linking asan_globals_win.cc into the DSO.
#define ASAN_LINK_GLOBALS_WIN() \
  __pragma(                     \
      comment(linker, "/include:" SANITIZER_SYM_PREFIX "__asan_dso_reg_hook"))

#endif // ASAN_GLOBALS_WIN_H

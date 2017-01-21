//===-- sanitizer_win_defs.h ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Common definitions for Windows-specific code.
//
//===----------------------------------------------------------------------===//
#ifndef SANITIZER_WIN_DEFS_H
#define SANITIZER_WIN_DEFS_H

#include "sanitizer_platform.h"
#if SANITIZER_WINDOWS

#if defined(_WIN64)
#define WIN_SYM_PREFIX
#else
#define WIN_SYM_PREFIX "_"
#endif

// Intermediate macro to ensure the parameter is expanded before stringified.
#define STRINGIFY(A) #A

// ----------------- A workaround for the absence of weak symbols --------------
// We don't have a direct equivalent of weak symbols when using MSVC, but we can
// use the /alternatename directive to tell the linker to default a specific
// symbol to a specific value.
// Take into account that the function will be marked as UNDEF in the symbol
// table of the resulting object file, even if we provided a default value, and
// the linker won't find the default implementation until it links with that
// object file.
// So, suppose we provide a default implementation "fundef" for "fun", and this
// is compiled into the object file "test.obj".
// If we have some code with references to "fun" and we link that code with
// "test.obj", it will work because the linker always link object files.
// But, if "test.obj" is included in a static library, like "test.lib", then the
// liker will only link to "test.obj" if necessary. If we only included the
// definition of "fun", it won't link to "test.obj" (from test.lib) because
// "fun" appears as UNDEF, so it doesn't resolve the symbol "fun", and this will
// result in a link error.
// So, a workaround is to force linkage with the modules that include weak
// definitions, with the following macro: WIN_FORCE_LINK()

#define WIN_WEAK_ALIAS_(Name, Default)                                         \
  __pragma(comment(linker, "/alternatename:" WIN_SYM_PREFIX STRINGIFY(Name) "="\
                                             WIN_SYM_PREFIX STRINGIFY(Default)))

#define WIN_WEAK_ALIAS(Name, Default)                                          \
  WIN_WEAK_ALIAS_(Name, Default)

#define WIN_FORCE_LINK(Name)                                                   \
  __pragma(comment(linker, "/include:" WIN_SYM_PREFIX STRINGIFY(Name)))

#endif // SANITIZER_WINDOWS
#endif // SANITIZER_WIN_DEFS_H

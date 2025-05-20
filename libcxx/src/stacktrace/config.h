//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_STACKTRACE_CONFIG_H
#define _LIBCPP_STACKTRACE_CONFIG_H

#include <__config>
#include <__config_site>

// Check for unwind.h -- could exist on any OS (in theory), but it (or `libunwind`) is likely on Linux systems, and also
// comes with XCode tools on MacOS.
#if __has_include(<unwind.h>)
#  define _LIBCPP_STACKTRACE_COLLECT_UNWIND
#endif

// For OSX specific stuff (generally controls whether we use `dyld`)
#if defined(__APPLE__)
#  define _LIBCPP_STACKTRACE_MACOS
#endif

// For Linux specific stuff (`link.h`, expanded functions in `dlfcn.h`, and ELF symtab parsing)
#if defined(__linux__)
#  define _LIBCPP_STACKTRACE_LINUX
#endif

// Whether we can invoke external processes via `posix_spawn`
#if __has_include(<spawn.h>) && _LIBCPP_STACKTRACE_ALLOW_TOOLS_AT_RUNTIME
#  define _LIBCPP_STACKTRACE_CAN_SPAWN_TOOLS
#endif

#if defined(_LIBCPP_WIN32API)
#  define _LIBCPP_STACKTRACE_WINDOWS
#endif

#endif // _LIBCPP_STACKTRACE_CONFIG_H

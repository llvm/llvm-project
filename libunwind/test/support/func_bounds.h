// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for locating the address range of a function.
//
// On MachO targets the linker synthesises section$start$SEG$sect /
// section$end$SEG$sect symbols; __asm() is used to bind them to valid C
// identifiers without the leading '_' implied by the Darwin User Label Prefix.
//
// On ELF targets the linker synthesises __start_<section> / __stop_<section>
// symbols for any section whose name is a valid C identifier.
// We don't use dladdr() because on musl it's a no-op when statically linked.

#ifndef LIBUNWIND_TEST_SUPPORT_FUNC_BOUNDS_H
#define LIBUNWIND_TEST_SUPPORT_FUNC_BOUNDS_H

#ifdef __APPLE__
#define FUNC_BOUNDS_DECL(name)                                                 \
  extern char name##_start __asm("section$start$__TEXT$__" #name);             \
  extern char name##_end __asm("section$end$__TEXT$__" #name)
#define FUNC_ATTR(name)                                                        \
  __attribute__((section("__TEXT,__" #name ",regular,pure_instructions")))
#define FUNC_START(name) (&name##_start)
#define FUNC_END(name) (&name##_end)
#else
#define FUNC_BOUNDS_DECL(name)                                                 \
  extern char __start_##name;                                                  \
  extern char __stop_##name
#define FUNC_ATTR(name) __attribute__((section(#name)))
#define FUNC_START(name) (&__start_##name)
#define FUNC_END(name) (&__stop_##name)
#endif

#endif // LIBUNWIND_TEST_SUPPORT_FUNC_BOUNDS_H

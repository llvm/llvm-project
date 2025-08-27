//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Test that _LIBCPP_FREESTANDING is not defined when -ffreestanding is not passed
// to the compiler but defined when -ffreestanding is passed to the compiler.

// RUN: %{cxx} %{flags} %{compile_flags} -fsyntax-only %s
// RUN: %{cxx} %{flags} %{compile_flags} -fsyntax-only -ffreestanding -DFREESTANDING %s

#include <__config>

#if defined(FREESTANDING) != defined(_LIBCPP_FREESTANDING)
#error _LIBCPP_FREESTANDING should be defined in freestanding mode and not \
       defined in non-freestanding mode
#endif

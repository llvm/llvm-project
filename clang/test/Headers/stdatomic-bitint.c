// RUN: %clang_cc1 -std=c23 -ffreestanding -fsyntax-only -verify %s
//===-- stdatomic-bitint.c - Atomic _BitInt header names ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <stdatomic.h>

#ifdef ATOMIC_BITINT_LOCK_FREE
#error Clang must not define ATOMIC_BITINT_LOCK_FREE
#endif

// expected-error@+1 {{unknown type name 'atomic_bit_int'}}
atomic_bit_int unsupported_alias;

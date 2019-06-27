//===-- wrappers_c.cc -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "platform.h"

// Skip this compilation unit if compiled as part of Bionic.
#if !SCUDO_ANDROID || !_BIONIC

#include "allocator_config.h"
#include "wrappers_c.h"
#include "wrappers_c_checks.h"

#include <stdint.h>
#include <stdio.h>

static scudo::Allocator<scudo::Config> Allocator;
// Pointer to the static allocator so that the C++ wrappers can access it.
// Technically we could have a completely separated heap for C & C++ but in
// reality the amount of cross pollination between the two is staggering.
scudo::Allocator<scudo::Config> *AllocatorPtr = &Allocator;

extern "C" {

#define SCUDO_PREFIX(name) name
#define SCUDO_ALLOCATOR Allocator
#include "wrappers_c.inc"
#undef SCUDO_ALLOCATOR
#undef SCUDO_PREFIX

INTERFACE void __scudo_print_stats(void) { Allocator.printStats(); }

} // extern "C"

#endif // !SCUDO_ANDROID || !_BIONIC

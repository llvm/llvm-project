//===-- wrappers_c_bionic.cpp -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "platform.h"

// This is only used when compiled as part of Bionic.
#if SCUDO_ANDROID && _BIONIC

#include "allocator_config.h"
#include "wrappers_c.h"
#include "wrappers_c_checks.h"

#include <stdint.h>
#include <stdio.h>

static scudo::Allocator<scudo::AndroidConfig> Allocator;
static scudo::Allocator<scudo::AndroidSvelteConfig> SvelteAllocator;

extern "C" {

// Regular MallocDispatch definitions.
#define SCUDO_PREFIX(name) CONCATENATE(scudo_, name)
#define SCUDO_ALLOCATOR Allocator
#include "wrappers_c.inc"
#undef SCUDO_ALLOCATOR
#undef SCUDO_PREFIX

// Svelte MallocDispatch definitions.
#define SCUDO_PREFIX(name) CONCATENATE(scudo_svelte_, name)
#define SCUDO_ALLOCATOR SvelteAllocator
#include "wrappers_c.inc"
#undef SCUDO_ALLOCATOR
#undef SCUDO_PREFIX

// The following is the only function that will end up initializing both
// allocators, which will result in a slight increase in memory footprint.
INTERFACE void __scudo_print_stats(void) {
  Allocator.printStats();
  SvelteAllocator.printStats();
}

} // extern "C"

#endif // SCUDO_ANDROID && _BIONIC

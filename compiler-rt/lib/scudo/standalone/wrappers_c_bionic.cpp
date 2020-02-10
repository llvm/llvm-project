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

// Regular MallocDispatch definitions.
#define SCUDO_PREFIX(name) CONCATENATE(scudo_, name)
#define SCUDO_ALLOCATOR Allocator

extern "C" void SCUDO_PREFIX(malloc_postinit)();
static scudo::Allocator<scudo::AndroidConfig, SCUDO_PREFIX(malloc_postinit)>
    SCUDO_ALLOCATOR;

#include "wrappers_c.inc"

#undef SCUDO_ALLOCATOR
#undef SCUDO_PREFIX

// Svelte MallocDispatch definitions.
#define SCUDO_PREFIX(name) CONCATENATE(scudo_svelte_, name)
#define SCUDO_ALLOCATOR SvelteAllocator

extern "C" void SCUDO_PREFIX(malloc_postinit)();
static scudo::Allocator<scudo::AndroidSvelteConfig,
                        SCUDO_PREFIX(malloc_postinit)>
    SCUDO_ALLOCATOR;

#include "wrappers_c.inc"

#undef SCUDO_ALLOCATOR
#undef SCUDO_PREFIX

// TODO(kostyak): support both allocators.
INTERFACE void __scudo_print_stats(void) { Allocator.printStats(); }

#endif // SCUDO_ANDROID && _BIONIC

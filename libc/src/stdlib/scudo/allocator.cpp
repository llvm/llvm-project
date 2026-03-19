//===-- Shared Scudo Allocator State ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "scudo_allocator.h"

#if defined(LIBC_FULL_BUILD)
#include "src/__support/threads/fork_callbacks.h"
#else
#include <pthread.h>
#endif

namespace LIBC_NAMESPACE_DECL {

namespace {

void malloc_disable() { Allocator.disable(); }

void malloc_enable() { Allocator.enable(); }

} // namespace

SCUDO_REQUIRE_CONSTANT_INITIALIZATION
scudo::Allocator<scudo::Config, malloc_postinit> Allocator;

void malloc_postinit() {
  Allocator.initGwpAsan();
#if defined(LIBC_FULL_BUILD)
  (void)register_atfork_callbacks(malloc_disable, malloc_enable, malloc_enable);
#else
  (void)pthread_atfork(malloc_disable, malloc_enable, malloc_enable);
#endif
}

} // namespace LIBC_NAMESPACE_DECL

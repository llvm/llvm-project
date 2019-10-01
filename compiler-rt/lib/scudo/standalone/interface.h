//===-- interface.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SCUDO_INTERFACE_H_
#define SCUDO_INTERFACE_H_

#include "internal_defs.h"

extern "C" {

WEAK INTERFACE const char *__scudo_default_options();

// Post-allocation & pre-deallocation hooks.
// They must be thread-safe and not use heap related functions.
WEAK INTERFACE void __scudo_allocate_hook(void *ptr, size_t size);
WEAK INTERFACE void __scudo_deallocate_hook(void *ptr);

WEAK INTERFACE void __scudo_print_stats(void);

typedef void (*iterate_callback)(uintptr_t base, size_t size, void *arg);

} // extern "C"

#endif // SCUDO_INTERFACE_H_

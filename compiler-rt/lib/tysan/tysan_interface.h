//===-- tysan_interface.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of TypeSanitizer.
//
// The functions declared in this header will be inserted by the instrumentation
// module.
// This header can be included by the instrumented program or by TySan tests.
//===----------------------------------------------------------------------===//

#ifndef TYSAN_INTERFACE_H
#define TYSAN_INTERFACE_H

#include <sanitizer_common/sanitizer_internal_defs.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

SANITIZER_INTERFACE_ATTRIBUTE
void __tysan_copy_shadow(const void *dst, const void *src, size_t type_size);

SANITIZER_INTERFACE_ATTRIBUTE
void __tysan_copy_shadow_array(const void *dst_array, const void *src,
                               size_t type_size, size_t arraySize);

SANITIZER_INTERFACE_ATTRIBUTE
void __tysan_reset_shadow(const void *addr, size_t size);

SANITIZER_INTERFACE_ATTRIBUTE
int __tysan_get_type_name(const void *addr, char *buffer, size_t buffer_size);

#ifdef __cplusplus
} // extern "C"
#endif

#endif

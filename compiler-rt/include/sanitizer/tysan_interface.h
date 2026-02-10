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
// Public interface header for TySan.
//===----------------------------------------------------------------------===//
#ifndef SANITIZER_TYSAN_INTERFACE_H
#define SANITIZER_TYSAN_INTERFACE_H

#include <sanitizer/common_interface_defs.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

// Copies the shadow memory for the source user memory into the shadow memory
// for the destination user memory
void SANITIZER_CDECL __tysan_copy_shadow(const void *dst, const void *src,
                                         size_t type_size);

// Copies the shadow memory for the source user memory into the shadow memory
// for each element in the destination array in user memory
void SANITIZER_CDECL __tysan_copy_shadow_array(const void *dst_array,
                                               const void *src,
                                               size_t type_size,
                                               size_t arraySize);

// Clears the shadow memory for the given range of user memory.
void SANITIZER_CDECL __tysan_reset_shadow(const void *addr, size_t size);

// Writes the name of the type represented in the shadow memory for the given
// location in user memory into the given buffer, up to the given size. Returns
// the length written.
int SANITIZER_CDECL __tysan_get_type_name(const void *addr, char *buffer,
                                          size_t buffer_size);

#ifdef __cplusplus
}
#endif

#endif

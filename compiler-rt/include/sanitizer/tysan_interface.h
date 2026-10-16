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

/// Marks a memory region (<c>[addr, addr+size)</c>) as not yet having a type.
///
/// TySan will take the next read/write to the memory region to be the correct
/// type for the memory, and use that for its checks from then on.
///
/// \param addr Start of memory region.
/// \param size Size of memory region.
void SANITIZER_CDECL __tysan_set_type_unknown(void const *addr, size_t size);

#ifdef __cplusplus
}
#endif

#endif

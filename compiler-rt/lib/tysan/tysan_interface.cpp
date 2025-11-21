//===-- tysan_interface.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of TypeSanitizer.
//
//===----------------------------------------------------------------------===//
#include "tysan_interface.h"
#include "tysan.h"

void __tysan_copy_shadow(const void *dst, const void *src, size_t type_size) {
  tysan_copy_types(dst, src, type_size);
}

void __tysan_copy_shadow_array(const void *dst_array, const void *src,
                               size_t type_size, size_t arraySize) {
  const void *dst = dst_array;
  for (size_t i = 0; i < arraySize; i++) {
    tysan_copy_types(dst, src, type_size);
    dst = (void *)(((uptr)dst) + type_size);
  }
}

void __tysan_reset_shadow(const void *addr, size_t size) {
  tysan_set_type_unknown(addr, size);
}

int __tysan_get_type_name(const void *addr, char *buffer, size_t buffer_size) {
  void **shadow = (void **)__tysan::shadow_for(addr);
  return getTDName(*shadow, buffer, buffer_size, false);
}

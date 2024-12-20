//===-- sanitizer_contiguous_container.cpp
//-----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
//
// This file provides weak defs of __sanitizer*contiguous_container* functions
// whose strong implementations can be defined in particular runtime libs
// of sanitizers
//
//===---------------------------------------------------------------------===//

#include "sanitizer_internal_defs.h"

SANITIZER_INTERFACE_WEAK_DEF(void, __sanitizer_annotate_contiguous_container,
                             const void *, const void *, const void *,
                             const void *) {}

SANITIZER_INTERFACE_WEAK_DEF(
    void, __sanitizer_annotate_double_ended_contiguous_container, const void *,
    const void *, const void *, const void *, const void *, const void *) {}

SANITIZER_INTERFACE_WEAK_DEF(void,
                             __sanitizer_copy_contiguous_container_annotations,
                             const void *, const void *, const void *,
                             const void *) {}

SANITIZER_INTERFACE_WEAK_DEF(int, __sanitizer_verify_contiguous_container,
                             const void *, const void *, const void *) {
  return 0;
}

SANITIZER_INTERFACE_WEAK_DEF(
    int, __sanitizer_verify_double_ended_contiguous_container, const void *,
    const void *, const void *, const void *) {
  return 0;
}

SANITIZER_INTERFACE_WEAK_DEF(const void *,
                             __sanitizer_contiguous_container_find_bad_address,
                             const void *, const void *, const void *) {
  return nullptr;
}

SANITIZER_INTERFACE_WEAK_DEF(
    const void *,
    __sanitizer_double_ended_contiguous_container_find_bad_address,
    const void *, const void *, const void *, const void *) {
  return nullptr;
}

//===-- ubsan_offload_preinit.cpp -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Call __ubsan_offload_init at the very early stage of process startup.
//
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_internal_defs.h"
#include "ubsan_offload.h"

#if SANITIZER_CAN_USE_PREINIT_ARRAY
// This section is linked into the main executable when offloading uses UBSan
// to perform initialization at a very early stage.
__attribute__((section(".preinit_array"), used)) static auto preinit =
    __ubsan_offload_init;
#endif

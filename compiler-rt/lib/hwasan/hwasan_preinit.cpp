//===-- hwasan_preinit.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of HWAddressSanitizer, an address sanity checker.
//
// Call __hwasan_init at the very early stage of process startup.
//===----------------------------------------------------------------------===//
#include "hwasan_interface_internal.h"
#include "sanitizer_common/sanitizer_internal_defs.h"

#if SANITIZER_CAN_USE_PREINIT_ARRAY
// The symbol is called __local_hwasan_preinit, because it's not intended to
// be exported.
// This code linked into the main executable when -fsanitize=hwaddress is in
// the link flags. It can only use exported interface functions.
__attribute__((section(".preinit_array"), used)) static void (
    *__local_hwasan_preinit)(void) = __hwasan_init;
#endif

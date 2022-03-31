//===-- hwasan_memintrinsics.cpp --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file is a part of HWAddressSanitizer and contains HWASAN versions of
/// memset, memcpy and memmove
///
//===----------------------------------------------------------------------===//

#include <string.h>
#include "hwasan.h"
#include "hwasan_checks.h"
#include "hwasan_flags.h"
#include "hwasan_interface_internal.h"
#include "sanitizer_common/sanitizer_libc.h"

using namespace __hwasan;

void *__hwasan_memset(void *block, int c, uptr size) {
  CheckAddressSized<ErrorAction::Recover, AccessType::Store>(
      reinterpret_cast<uptr>(block), size);
  return memset(block, c, size);
}

void *__hwasan_memcpy(void *to, const void *from, uptr size) {
  CheckAddressSized<ErrorAction::Recover, AccessType::Store>(
      reinterpret_cast<uptr>(to), size);
  CheckAddressSized<ErrorAction::Recover, AccessType::Load>(
      reinterpret_cast<uptr>(from), size);
  return memcpy(to, from, size);
}

void *__hwasan_memmove(void *to, const void *from, uptr size) {
  CheckAddressSized<ErrorAction::Recover, AccessType::Store>(
      reinterpret_cast<uptr>(to), size);
  CheckAddressSized<ErrorAction::Recover, AccessType::Load>(
      reinterpret_cast<uptr>(from), size);
  return memmove(to, from, size);
}

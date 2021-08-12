//===-- scudo_utils.h -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Header for scudo_utils.cpp.
///
//===----------------------------------------------------------------------===//

#ifndef SCUDO_UTILS_H_
#define SCUDO_UTILS_H_

#include "sanitizer_common/sanitizer_common.h"

#include <string.h>

namespace __scudo {

template <class Dest, class Source>
inline Dest bit_cast(const Source& source) {
  static_assert(sizeof(Dest) == sizeof(Source), "Sizes are not equal!");
  Dest dest;
  memcpy(&dest, &source, sizeof(dest));
  return dest;
}

void dieWithMessage(const char *Format, ...) NORETURN FORMAT(1, 2);

bool hasHardwareCRC32();

}  // namespace __scudo

#endif  // SCUDO_UTILS_H_

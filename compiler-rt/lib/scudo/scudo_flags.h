//===-- scudo_flags.h -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Header for scudo_flags.cpp.
///
//===----------------------------------------------------------------------===//

#ifndef SCUDO_FLAGS_H_
#define SCUDO_FLAGS_H_

namespace __scudo {

struct Flags {
#define SCUDO_FLAG(Type, Name, DefaultValue, Description) Type Name;
#include "scudo_flags.inc"
#undef SCUDO_FLAG

  void setDefaults();
};

Flags *getFlags();

void initFlags();

}  // namespace __scudo

#endif  // SCUDO_FLAGS_H_

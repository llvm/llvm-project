//===-- copyprof_flags.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file declares the configuration flags structure and initialization
/// interface for the CopyProf runtime library.
///
//===----------------------------------------------------------------------===//

#ifndef COPYPROF_FLAGS_H_
#define COPYPROF_FLAGS_H_

namespace __copyprof {

struct Flags {
#define COPYPROF_FLAG(Type, Name, DefaultValue, Description) Type Name;
#include "copyprof_flags.inc"
#undef COPYPROF_FLAG

  void SetDefaults();
};

Flags* flags();

void InitializeFlags();

}  // namespace __copyprof

#endif  // COPYPROF_FLAGS_H_

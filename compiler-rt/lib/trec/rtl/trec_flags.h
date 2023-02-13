//===-- trec_flags.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of TraceRecorder (TRec), a race detector.
// NOTE: This file may be included into user code.
//===----------------------------------------------------------------------===//

#ifndef TREC_FLAGS_H
#define TREC_FLAGS_H

#include "sanitizer_common/sanitizer_flags.h"

namespace __trec {

struct Flags {
#define TREC_FLAG(Type, Name, DefaultValue, Description) Type Name;
#include "trec_flags.inc"
#undef TREC_FLAG

  void SetDefaults();
};

void InitializeFlags(Flags *flags, const char *env,
                     const char *env_option_name = nullptr);
}  // namespace __trec

#endif  // TREC_FLAGS_H

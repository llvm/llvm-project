//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Device address sanitizer runtime flags.
///
//===----------------------------------------------------------------------===//

#ifndef DASAN_FLAGS_H
#define DASAN_FLAGS_H

#include "sanitizer_common/sanitizer_internal_defs.h"

namespace __dasan {

struct Flags {
#define DASAN_FLAG(Type, Name, Default, Description) Type Name;
#include "dasan_flags.inc"
#undef DASAN_FLAG

  void SetDefaults();
};

extern Flags DasanFlags;
inline Flags* flags() { return &DasanFlags; }

void InitializeFlags();

}  // namespace __dasan

#endif  // DASAN_FLAGS_H

//===- MutagenUtilPosix.cpp - Misc utils for Posix. -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Misc utils implementation using Posix API.
//===----------------------------------------------------------------------===//
#include "FuzzerPlatform.h"
#if (LIBFUZZER_POSIX || LIBFUZZER_FUCHSIA)
#include <cstring>

namespace mutagen {

const void *SearchMemory(const void *Data, size_t DataLen, const void *Patt,
                         size_t PattLen) {
  return memmem(Data, DataLen, Patt, PattLen);
}

} // namespace mutagen

#endif // (LIBFUZZER_POSIX || LIBFUZZER_FUCHSIA)

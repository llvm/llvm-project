//===- MutagenUtilWindows.cpp - Misc utils for Windows. -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Misc utils implementation for Windows.
//===----------------------------------------------------------------------===//
#include "FuzzerPlatform.h"
#if LIBFUZZER_WINDOWS
#include <cstddef>
#include <cstdio>
#include <cstring>

namespace mutagen {

const void *SearchMemory(const void *Data, size_t DataLen, const void *Patt,
                         size_t PattLen) {
  // TODO: make this implementation more efficient.
  const char *Cdata = (const char *)Data;
  const char *Cpatt = (const char *)Patt;

  if (!Data || !Patt || DataLen == 0 || PattLen == 0 || DataLen < PattLen)
    return NULL;

  if (PattLen == 1)
    return memchr(Data, *Cpatt, DataLen);

  const char *End = Cdata + DataLen - PattLen + 1;

  for (const char *It = Cdata; It < End; ++It)
    if (It[0] == Cpatt[0] && memcmp(It, Cpatt, PattLen) == 0)
      return It;

  return NULL;
}

} // namespace mutagen

#endif // LIBFUZZER_WINDOWS

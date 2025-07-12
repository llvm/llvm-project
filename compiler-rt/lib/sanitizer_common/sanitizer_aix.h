//===-- sanitizer_aix.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is shared between various sanitizers' runtime libraries and
// provides definitions for AIX-specific functions.
//===----------------------------------------------------------------------===//
#ifndef SANITIZER_AIX_H
#define SANITIZER_AIX_H

#include "sanitizer_platform.h"

#if SANITIZER_AIX
#  include "sanitizer_common.h"
#  include "sanitizer_posix.h"

struct prmap;
typedef struct prmap prmap_t;

namespace __sanitizer {

struct ProcSelfMapsBuff {
  char *data;
  uptr mmaped_size;
  uptr len;
  prmap_t *mapEnd;
};

struct MemoryMappingLayoutData {
  ProcSelfMapsBuff proc_self_maps;
  const char *current;
};

void ReadProcMaps(ProcSelfMapsBuff *proc_maps);

}  // namespace __sanitizer

#endif  // SANITIZER_AIX
#endif  // SANITIZER_AIX_H

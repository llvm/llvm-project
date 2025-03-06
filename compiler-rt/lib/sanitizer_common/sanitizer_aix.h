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
#include "sanitizer_common.h"
#  include "sanitizer_posix.h"

namespace __sanitizer {

#if SANITIZER_WORDSIZE == 32
static const uptr InstructionStart = 0x10000000;
#else
static const uptr InstructionStart = 0x100000000;
#endif

struct ProcSelfMapsBuff {
  char *data;
  uptr mmaped_size;
  uptr len;
};

struct MemoryMappingLayoutData {
  ProcSelfMapsBuff proc_self_maps;
  const char *current;
};

void ReadProcMaps(ProcSelfMapsBuff *proc_maps);

char *internal_getcwd(char *buf, uptr size);

}  // namespace __sanitizer

#endif  // SANITIZER_AIX
#endif  // SANITIZER_AIX_H

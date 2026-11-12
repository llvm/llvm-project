//===- ScriptParser.h -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_ELF_SCRIPT_PARSER_H
#define LLD_ELF_SCRIPT_PARSER_H

#include "lld/Common/LLVM.h"
#include "llvm/Support/MemoryBufferRef.h"

namespace lld::elf {
struct Ctx;

// Parses a linker script. Calling this function updates
// lld::elf::config and lld::elf::script.
void readLinkerScript(Ctx &ctx, MemoryBufferRef mb);

// Parses a version script.
void readVersionScript(Ctx &ctx, MemoryBufferRef mb);

void readDynamicList(Ctx &ctx, MemoryBufferRef mb);

// Parses the defsym expression.
void readDefsym(Ctx &ctx, MemoryBufferRef mb);

bool hasWildcard(StringRef s);

} // namespace lld::elf

#endif

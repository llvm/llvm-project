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
#include "llvm/Support/MemoryBuffer.h"

namespace lld {
namespace elf {

// Parses a linker script. Calling this function updates
// Config and ScriptConfig.
void readLinkerScript(MemoryBufferRef MB);

// Parses a version script.
void readVersionScript(MemoryBufferRef MB);

void readDynamicList(MemoryBufferRef MB);

// Parses the defsym expression.
void readDefsym(StringRef Name, MemoryBufferRef MB);

} // namespace elf
} // namespace lld

#endif

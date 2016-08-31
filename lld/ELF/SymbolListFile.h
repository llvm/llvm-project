//===- SymbolListFile.h -----------------------------------------*- C++ -*-===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_ELF_SYMBOL_LIST_FILE_H
#define LLD_ELF_SYMBOL_LIST_FILE_H

#include "lld/Core/LLVM.h"
#include "llvm/Support/MemoryBuffer.h"

namespace lld {
namespace elf {

void parseDynamicList(MemoryBufferRef MB);

} // namespace elf
} // namespace lld

#endif

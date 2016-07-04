//===- Strings.h ------------------------------------------------*- C++ -*-===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_COFF_STRINGS_H
#define LLD_COFF_STRINGS_H

#include "lld/Core/LLVM.h"
#include <vector>

namespace lld {
namespace elf {
bool globMatch(StringRef S, StringRef T);
std::vector<uint8_t> parseHex(StringRef S);
bool isValidCIdentifier(StringRef S);
}
}

#endif

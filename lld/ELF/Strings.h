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
#include "llvm/Support/Regex.h"
#include <vector>

namespace lld {
namespace elf {
llvm::Regex compileGlobPatterns(ArrayRef<StringRef> V);
int getPriority(StringRef S);
bool hasWildcard(StringRef S);
std::vector<uint8_t> parseHex(StringRef S);
bool isValidCIdentifier(StringRef S);
StringRef unquote(StringRef S);

// Returns a demangled C++ symbol name. If Name is not a mangled
// name or the system does not provide __cxa_demangle function,
// it returns an unmodified string.
std::string demangle(StringRef Name);
}
}

#endif

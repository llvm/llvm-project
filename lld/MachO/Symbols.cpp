//===- Symbols.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Symbols.h"
#include "InputFiles.h"
#include "lld/Common/ErrorHandler.h"
#include "lld/Common/Strings.h"

using namespace llvm;
using namespace lld;
using namespace lld::macho;

// Returns a symbol for an error message.
std::string lld::toString(const Symbol &sym) {
  if (Optional<std::string> s = demangleItanium(sym.getName()))
    return *s;
  return std::string(sym.getName());
}

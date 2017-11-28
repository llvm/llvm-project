//===- Strings.cpp -------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Strings.h"
#include "Config.h"
#include "lld/Common/Strings.h"
#include "llvm/ADT/StringRef.h"

using namespace llvm;

std::string lld::wasm::displayName(StringRef Name) {
  if (Config->Demangle)
    if (Optional<std::string> S = demangleItanium(Name))
      return "`" + *S + "'";
  return Name;
}

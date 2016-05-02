//===-- InMemoryXrefsDB.cpp -----------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "InMemoryXrefsDB.h"

namespace clang {
namespace include_fixer {

std::vector<std::string> InMemoryXrefsDB::search(llvm::StringRef Identifier) {
  auto I = LookupTable.find(Identifier);
  if (I != LookupTable.end())
    return I->second;
  return {};
}

} // namespace include_fixer
} // namespace clang

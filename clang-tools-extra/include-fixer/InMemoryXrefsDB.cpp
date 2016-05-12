//===-- InMemoryXrefsDB.cpp -----------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "InMemoryXrefsDB.h"

using clang::find_all_symbols::SymbolInfo;

namespace clang {
namespace include_fixer {

InMemoryXrefsDB::InMemoryXrefsDB(
    const std::map<std::string, std::vector<std::string>> &LookupTable) {
  for (const auto &Entry : LookupTable) {
    llvm::StringRef Identifier(Entry.first);
    llvm::SmallVector<llvm::StringRef, 8> Names;
    Identifier.split(Names, "::");
    for (const auto &Header : Entry.second) {
      std::vector<SymbolInfo::Context> Contexts;
      for (auto IdentiferContext = Names.rbegin() + 1;
           IdentiferContext != Names.rend(); ++IdentiferContext) {
        Contexts.emplace_back(SymbolInfo::ContextType::Namespace,
                              *IdentiferContext);
      }

      SymbolInfo Symbol(Names.back(), SymbolInfo::Class, Header, Contexts, 1);
      this->LookupTable[Symbol.getName()].push_back(Symbol);
    }
  }
}

std::vector<SymbolInfo> InMemoryXrefsDB::search(llvm::StringRef Identifier) {
  auto I = LookupTable.find(Identifier);
  if (I != LookupTable.end())
    return I->second;
  return {};
}

} // namespace include_fixer
} // namespace clang

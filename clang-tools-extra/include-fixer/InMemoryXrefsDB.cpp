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
      // FIXME: create a complete instance with static member function when it
      // is implemented.
      SymbolInfo Info;
      Info.Name = Names.back();
      Info.FilePath = Header;
      for (auto IdentiferContext = Names.rbegin() + 1;
           IdentiferContext != Names.rend(); ++IdentiferContext) {
        Info.Contexts.push_back(
            {SymbolInfo::ContextType::Namespace, *IdentiferContext});
      }
      this->LookupTable[Info.Name].push_back(Info);
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

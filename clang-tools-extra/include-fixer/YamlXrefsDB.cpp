//===-- YamlXrefsDB.cpp ---------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "YamlXrefsDB.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include <string>
#include <vector>

namespace clang {
namespace include_fixer {

YamlXrefsDB::YamlXrefsDB(llvm::StringRef FilePath) {
  int ReadFD = 0;
  if (llvm::sys::fs::openFileForRead(FilePath, ReadFD))
    return;
  auto Buffer = llvm::MemoryBuffer::getOpenFile(ReadFD, FilePath, -1);
  if (!Buffer)
    return;
  Symbols = clang::find_all_symbols::ReadSymbolInfosFromYAML(
      Buffer.get()->getBuffer());
}

std::vector<std::string> YamlXrefsDB::search(llvm::StringRef Identifier) {
  llvm::SmallVector<llvm::StringRef, 16> Names;
  std::vector<std::string> Results;

  // The identifier may be fully qualified, so split it and get all the context
  // names.
  Identifier.split(Names, "::");
  for (const auto &Symbol : Symbols) {
    // Match the identifier name without qualifier.
    if (Symbol.Name == Names.back()) {
      bool IsMatched = true;
      auto SymbolContext = Symbol.Contexts.begin();
      // Match the remaining context names.
      for (auto IdentiferContext = Names.rbegin() + 1;
           IdentiferContext != Names.rend() &&
           SymbolContext != Symbol.Contexts.end();
           ++IdentiferContext, ++SymbolContext) {
        if (SymbolContext->second != *IdentiferContext) {
          IsMatched = false;
          break;
        }
      }

      if (IsMatched) {
        Results.push_back("\"" + Symbol.FilePath + "\"");
      }
    }
  }
  return Results;
}

} // namespace include_fixer
} // namespace clang

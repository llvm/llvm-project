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

using clang::find_all_symbols::SymbolInfo;

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

std::vector<SymbolInfo> YamlXrefsDB::search(llvm::StringRef Identifier) {
  std::vector<SymbolInfo> Results;
  for (const auto &Symbol : Symbols) {
    if (Symbol.Name == Identifier)
      Results.push_back(Symbol);
  }
  return Results;
}

} // namespace include_fixer
} // namespace clang

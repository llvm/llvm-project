//===-- YamlSymbolIndex.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "YamlSymbolIndex.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include <string>
#include <vector>

using clang::find_all_symbols::SymbolInfo;
using clang::find_all_symbols::SymbolAndSignals;

namespace clang {
namespace include_fixer {

llvm::ErrorOr<std::unique_ptr<YamlSymbolIndex>>
YamlSymbolIndex::createFromFile(llvm::StringRef FilePath) {
  auto Buffer = llvm::MemoryBuffer::getFile(FilePath);
  if (!Buffer)
    return Buffer.getError();

  return std::unique_ptr<YamlSymbolIndex>(new YamlSymbolIndex(
      find_all_symbols::ReadSymbolInfosFromYAML(Buffer.get()->getBuffer())));
}

llvm::ErrorOr<std::unique_ptr<YamlSymbolIndex>>
YamlSymbolIndex::createFromDirectory(llvm::StringRef Directory,
                                     llvm::StringRef Name) {
  // Walk upwards from Directory, looking for files.
  for (llvm::SmallString<128> PathStorage = Directory; !Directory.empty();
       Directory = llvm::sys::path::parent_path(Directory)) {
    assert(Directory.size() <= PathStorage.size());
    PathStorage.resize(Directory.size()); // Shrink to parent.
    llvm::sys::path::append(PathStorage, Name);
    if (auto DB = createFromFile(PathStorage))
      return DB;
  }
  return llvm::make_error_code(llvm::errc::no_such_file_or_directory);
}

std::vector<SymbolAndSignals>
YamlSymbolIndex::search(llvm::StringRef Identifier) {
  std::vector<SymbolAndSignals> Results;
  for (const auto &Symbol : Symbols) {
    if (Symbol.Symbol.getName() == Identifier)
      Results.push_back(Symbol);
  }
  return Results;
}

} // namespace include_fixer
} // namespace clang

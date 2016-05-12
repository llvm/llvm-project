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
#include "llvm/Support/Errc.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include <string>
#include <vector>

using clang::find_all_symbols::SymbolInfo;

namespace clang {
namespace include_fixer {

llvm::ErrorOr<std::unique_ptr<YamlXrefsDB>>
YamlXrefsDB::createFromFile(llvm::StringRef FilePath) {
  auto Buffer = llvm::MemoryBuffer::getFile(FilePath);
  if (!Buffer)
    return Buffer.getError();

  return std::unique_ptr<YamlXrefsDB>(
      new YamlXrefsDB(clang::find_all_symbols::ReadSymbolInfosFromYAML(
          Buffer.get()->getBuffer())));
}

llvm::ErrorOr<std::unique_ptr<YamlXrefsDB>>
YamlXrefsDB::createFromDirectory(llvm::StringRef Directory,
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

std::vector<SymbolInfo> YamlXrefsDB::search(llvm::StringRef Identifier) {
  std::vector<SymbolInfo> Results;
  for (const auto &Symbol : Symbols) {
    if (Symbol.getName() == Identifier)
      Results.push_back(Symbol);
  }
  return Results;
}

} // namespace include_fixer
} // namespace clang

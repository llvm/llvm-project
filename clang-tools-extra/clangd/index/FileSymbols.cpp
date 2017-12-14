//===--- FileSymbols.cpp - Symbols from files. ------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "FileSymbols.h"
#include "clang/Index/IndexingAction.h"

namespace clang {
namespace clangd {

void FileSymbols::update(PathRef Path, std::unique_ptr<SymbolSlab> Slab) {
  std::lock_guard<std::mutex> Lock(Mutex);
  if (!Slab)
    FileToSlabs.erase(Path);
  else
    FileToSlabs[Path] = std::shared_ptr<SymbolSlab>(Slab.release());
}

std::shared_ptr<std::vector<const Symbol *>> FileSymbols::allSymbols() {
  // The snapshot manages life time of symbol slabs and provides pointers of all
  // symbols in all slabs.
  struct Snapshot {
    std::vector<const Symbol *> Pointers;
    std::vector<std::shared_ptr<SymbolSlab>> KeepAlive;
  };
  auto Snap = std::make_shared<Snapshot>();
  {
    std::lock_guard<std::mutex> Lock(Mutex);

    for (const auto &FileAndSlab : FileToSlabs) {
      Snap->KeepAlive.push_back(FileAndSlab.second);
      for (const auto &Iter : *FileAndSlab.second)
        Snap->Pointers.push_back(&Iter.second);
    }
  }
  auto *Pointers = &Snap->Pointers;
  // Use aliasing constructor to keep the snapshot alive along with the
  // pointers.
  return {std::move(Snap), Pointers};
}

} // namespace clangd
} // namespace clang

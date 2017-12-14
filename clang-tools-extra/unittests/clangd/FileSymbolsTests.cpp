//===-- FileSymbolsTests.cpp  -------------------------*- C++ -*-----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "index/FileSymbols.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using testing::UnorderedElementsAre;

namespace clang {
namespace clangd {

namespace {

Symbol symbol(llvm::StringRef ID) {
  Symbol Sym;
  Sym.ID = SymbolID(ID);
  Sym.QualifiedName = ID;
  return Sym;
}

void addNumSymbolsToSlab(int Begin, int End, SymbolSlab *Slab) {
  for (int i = Begin; i <= End; i++)
    Slab->insert(symbol(std::to_string(i)));
}

std::vector<std::string>
getSymbolNames(const std::vector<const Symbol *> &Symbols) {
  std::vector<std::string> Names;
  for (const Symbol *Sym : Symbols)
    Names.push_back(Sym->QualifiedName);
  return Names;
}

TEST(FileSymbolsTest, UpdateAndGet) {
  FileSymbols FS;
  EXPECT_THAT(getSymbolNames(*FS.allSymbols()), UnorderedElementsAre());

  auto Slab = llvm::make_unique<SymbolSlab>();
  addNumSymbolsToSlab(1, 3, Slab.get());

  FS.update("f1", std::move(Slab));

  EXPECT_THAT(getSymbolNames(*FS.allSymbols()),
              UnorderedElementsAre("1", "2", "3"));
}

TEST(FileSymbolsTest, Overlap) {
  FileSymbols FS;

  auto Slab = llvm::make_unique<SymbolSlab>();
  addNumSymbolsToSlab(1, 3, Slab.get());

  FS.update("f1", std::move(Slab));

  Slab = llvm::make_unique<SymbolSlab>();
  addNumSymbolsToSlab(3, 5, Slab.get());

  FS.update("f2", std::move(Slab));

  EXPECT_THAT(getSymbolNames(*FS.allSymbols()),
              UnorderedElementsAre("1", "2", "3", "3", "4", "5"));
}

TEST(FileSymbolsTest, SnapshotAliveAfterRemove) {
  FileSymbols FS;

  auto Slab = llvm::make_unique<SymbolSlab>();
  addNumSymbolsToSlab(1, 3, Slab.get());

  FS.update("f1", std::move(Slab));

  auto Symbols = FS.allSymbols();
  EXPECT_THAT(getSymbolNames(*Symbols), UnorderedElementsAre("1", "2", "3"));

  FS.update("f1", nullptr);
  EXPECT_THAT(getSymbolNames(*FS.allSymbols()), UnorderedElementsAre());

  EXPECT_THAT(getSymbolNames(*Symbols), UnorderedElementsAre("1", "2", "3"));
}

} // namespace
} // namespace clangd
} // namespace clang


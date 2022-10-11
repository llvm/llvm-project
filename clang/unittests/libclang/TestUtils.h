//===- unittests/libclang/TestUtils.h -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TEST_TESTUTILS_H
#define LLVM_CLANG_TEST_TESTUTILS_H

#include "clang-c/Index.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include "gtest/gtest.h"

class LibclangParseTest : public ::testing::Test {
  std::set<std::string> Files;
  typedef std::unique_ptr<std::string> fixed_addr_string;
  std::map<fixed_addr_string, fixed_addr_string> UnsavedFileContents;
public:
  std::string TestDir;
  CXIndex Index;
  CXTranslationUnit ClangTU;
  unsigned TUFlags;
  std::vector<CXUnsavedFile> UnsavedFiles;

  void SetUp() override {
    llvm::SmallString<256> Dir;
    ASSERT_FALSE(llvm::sys::fs::createUniqueDirectory("libclang-test", Dir));
    TestDir = std::string(Dir.str());
    TUFlags = CXTranslationUnit_DetailedPreprocessingRecord |
      clang_defaultEditingTranslationUnitOptions();
    Index = clang_createIndex(0, 0);
    ClangTU = nullptr;
  }
  void TearDown() override {
    clang_disposeTranslationUnit(ClangTU);
    clang_disposeIndex(Index);
    for (const std::string &Path : Files)
      llvm::sys::fs::remove(Path);
    llvm::sys::fs::remove(TestDir);
  }
  void WriteFile(std::string &Filename, const std::string &Contents) {
    if (!llvm::sys::path::is_absolute(Filename)) {
      llvm::SmallString<256> Path(TestDir);
      llvm::sys::path::append(Path, Filename);
      Filename = std::string(Path.str());
      Files.insert(Filename);
    }
    llvm::sys::fs::create_directories(llvm::sys::path::parent_path(Filename));
    std::ofstream OS(Filename);
    OS << Contents;
    assert(OS.good());
  }
  void MapUnsavedFile(std::string Filename, const std::string &Contents) {
    if (!llvm::sys::path::is_absolute(Filename)) {
      llvm::SmallString<256> Path(TestDir);
      llvm::sys::path::append(Path, Filename);
      Filename = std::string(Path.str());
    }
    auto it = UnsavedFileContents.insert(std::make_pair(
        fixed_addr_string(new std::string(Filename)),
        fixed_addr_string(new std::string(Contents))));
    UnsavedFiles.push_back({
        it.first->first->c_str(),   // filename
        it.first->second->c_str(),  // contents
        it.first->second->size()    // length
    });
  }
  template<typename F>
  void Traverse(const F &TraversalFunctor) {
    CXCursor TuCursor = clang_getTranslationUnitCursor(ClangTU);
    std::reference_wrapper<const F> FunctorRef = std::cref(TraversalFunctor);
    clang_visitChildren(TuCursor,
        &TraverseStateless<std::reference_wrapper<const F>>,
        &FunctorRef);
  }
  static std::string fromCXString(CXString cx_string) {
    std::string string{clang_getCString(cx_string)};
    clang_disposeString(cx_string);
    return string;
  };

private:
  template<typename TState>
  static CXChildVisitResult TraverseStateless(CXCursor cx, CXCursor parent,
      CXClientData data) {
    TState *State = static_cast<TState*>(data);
    return State->get()(cx, parent);
  }
};

#endif // LLVM_CLANG_TEST_TESTUTILS_H

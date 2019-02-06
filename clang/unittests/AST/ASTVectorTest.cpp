//===- unittests/AST/DeclTest.cpp --- Declaration tests -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unit tests for the ASTVector container.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTVector.h"
#include "clang/Basic/Builtins.h"
#include "gtest/gtest.h"

using namespace clang;

namespace clang {
namespace ast {

namespace {
class ASTVectorTest : public ::testing::Test {
protected:
  ASTVectorTest()
      : FileMgr(FileMgrOpts), DiagID(new DiagnosticIDs()),
        Diags(DiagID, new DiagnosticOptions, new IgnoringDiagConsumer()),
        SourceMgr(Diags, FileMgr), Idents(LangOpts, nullptr),
        Ctxt(LangOpts, SourceMgr, Idents, Sels, Builtins) {}

  FileSystemOptions FileMgrOpts;
  FileManager FileMgr;
  IntrusiveRefCntPtr<DiagnosticIDs> DiagID;
  DiagnosticsEngine Diags;
  SourceManager SourceMgr;
  LangOptions LangOpts;
  IdentifierTable Idents;
  SelectorTable Sels;
  Builtin::Context Builtins;
  ASTContext Ctxt;
};
} // unnamed namespace

TEST_F(ASTVectorTest, Compile) {
  ASTVector<int> V;
  V.insert(Ctxt, V.begin(), 0);
}

TEST_F(ASTVectorTest, InsertFill) {
  ASTVector<double> V;

  // Ensure returned iterator points to first of inserted elements
  auto I = V.insert(Ctxt, V.begin(), 5, 1.0);
  ASSERT_EQ(V.begin(), I);

  // Check non-empty case as well
  I = V.insert(Ctxt, V.begin() + 1, 5, 1.0);
  ASSERT_EQ(V.begin() + 1, I);

  // And insert-at-end
  I = V.insert(Ctxt, V.end(), 5, 1.0);
  ASSERT_EQ(V.end() - 5, I);
}

TEST_F(ASTVectorTest, InsertEmpty) {
  ASTVector<double> V;

  // Ensure no pointer overflow when inserting empty range
  int Values[] = { 0, 1, 2, 3 };
  ArrayRef<int> IntVec(Values);
  auto I = V.insert(Ctxt, V.begin(), IntVec.begin(), IntVec.begin());
  ASSERT_EQ(V.begin(), I);
  ASSERT_TRUE(V.empty());

  // Non-empty range
  I = V.insert(Ctxt, V.begin(), IntVec.begin(), IntVec.end());
  ASSERT_EQ(V.begin(), I);

  // Non-Empty Vector, empty range
  I = V.insert(Ctxt, V.end(), IntVec.begin(), IntVec.begin());
  ASSERT_EQ(V.begin() + IntVec.size(), I);

  // Non-Empty Vector, non-empty range
  I = V.insert(Ctxt, V.end(), IntVec.begin(), IntVec.end());
  ASSERT_EQ(V.begin() + IntVec.size(), I);
}

} // end namespace ast
} // end namespace clang

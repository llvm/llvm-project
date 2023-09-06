//===- unittests/AST/ASTDumperTest.cpp --- Test of AST node dump() methods ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains tests for TypeLoc::dump() and related methods.
// Most of these are lit tests via clang -ast-dump. However some nodes are not
// included in dumps of (TranslationUnit)Decl, but still relevant when dumped
// directly.
//
//===----------------------------------------------------------------------===//

#include "ASTPrint.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/Testing/TestAST.h"
#include "llvm/ADT/StringRef.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace clang;
using namespace ast_matchers;

namespace {
using testing::ElementsAre;
using testing::StartsWith;

std::vector<std::string> dumpTypeLoc(llvm::StringRef Name, ASTContext &Ctx) {
  auto Lookup = Ctx.getTranslationUnitDecl()->lookup(&Ctx.Idents.get(Name));
  auto *D = Lookup.find_first<DeclaratorDecl>();
  EXPECT_NE(D, nullptr) << Name;
  if (!D)
    return {};
  EXPECT_NE(D->getTypeSourceInfo(), nullptr);
  if (!D->getTypeSourceInfo())
    return {};
  std::string S;
  {
    llvm::raw_string_ostream OS(S);
    D->getTypeSourceInfo()->getTypeLoc().dump(OS, Ctx);
  }
  // Split result into lines.
  std::vector<std::string> Result;
  auto Remaining = llvm::StringRef(S).trim("\n");
  while (!Remaining.empty()) {
    auto [First, Rest] = Remaining.split('\n');
    Result.push_back(First.str());
    Remaining = Rest;
  }
  return Result;
}

TEST(ASTDumper, TypeLocChain) {
  TestAST AST(R"cc(
    const int **x;
  )cc");
  EXPECT_THAT(
      dumpTypeLoc("x", AST.context()),
      ElementsAre(""
                  "PointerTypeLoc <input.mm:2:11, col:16> 'const int **'",
                  "`-PointerTypeLoc <col:11, col:15> 'const int *'",
                  "  `-QualifiedTypeLoc <col:11> 'const int'",
                  "    `-BuiltinTypeLoc <col:11> 'int'"));
}

TEST(ASTDumper, FunctionTypeLoc) {
  TestAST AST(R"cc(
    void x(int, double *y);
  )cc");
  EXPECT_THAT(
      dumpTypeLoc("x", AST.context()),
      ElementsAre(""
                  "FunctionProtoTypeLoc <input.mm:2:5, col:26> 'void (int, "
                  "double *)' cdecl",
                  StartsWith("|-ParmVarDecl"),
                  "| `-BuiltinTypeLoc <col:12> 'int'",
                  StartsWith("|-ParmVarDecl"),
                  "| `-PointerTypeLoc <col:17, col:24> 'double *'",
                  "|   `-BuiltinTypeLoc <col:17> 'double'",
                  "`-BuiltinTypeLoc <col:5> 'void'"));
}

} // namespace
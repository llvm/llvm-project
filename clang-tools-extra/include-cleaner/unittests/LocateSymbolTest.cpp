//===--- LocateSymbolTest.cpp -------------------------------------- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "AnalysisInternal.h"
#include "clang-include-cleaner/Types.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Testing/TestAST.h"
#include "clang/Tooling/Inclusions/StandardLibrary.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Testing/Annotations/Annotations.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <cstddef>
#include <memory>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace clang::include_cleaner {
namespace {
using testing::ElementsAre;
using testing::ElementsAreArray;
using testing::Pair;
using testing::UnorderedElementsAre;

// A helper for building ASTs and getting decls out of it by name. Example usage
// looks like:
//   LocateExample X("void ^foo();");
//   Decl &Foo = X.findDecl("foo");
//   X.points(); // returns all the points in annotated test input.
struct LocateExample {
private:
  llvm::Annotations Target;
  TestAST AST;

public:
  LocateExample(llvm::StringRef AnnotatedCode)
      : Target(AnnotatedCode), AST([this] {
          TestInputs Inputs(Target.code());
          Inputs.ExtraArgs.push_back("-std=c++17");
          return Inputs;
        }()) {}

  const Decl &findDecl(llvm::StringRef SymbolName) {
    struct Visitor : RecursiveASTVisitor<Visitor> {
      llvm::StringRef NameToFind;
      const NamedDecl *Out = nullptr;
      bool VisitNamedDecl(const NamedDecl *ND) {
        if (ND->getName() == NameToFind) {
          EXPECT_TRUE(Out == nullptr || Out == ND->getCanonicalDecl())
              << "Found multiple matches for " << NameToFind;
          Out = cast<NamedDecl>(ND->getCanonicalDecl());
        }
        return true;
      }
    };
    Visitor V;
    V.NameToFind = SymbolName;
    V.TraverseDecl(AST.context().getTranslationUnitDecl());
    if (!V.Out)
      ADD_FAILURE() << "Couldn't find any decls with name: " << SymbolName;
    assert(V.Out);
    return *V.Out;
  }

  Macro findMacro(llvm::StringRef Name) {
    auto &PP = AST.preprocessor();
    auto *II = PP.getIdentifierInfo(Name);
    if (!II || !II->hasMacroDefinition()) {
      ADD_FAILURE() << "Couldn't find any macros with name: " << Name;
      return {};
    }
    auto MD = PP.getMacroDefinition(II);
    assert(MD.getMacroInfo());
    return {II, MD.getMacroInfo()->getDefinitionLoc()};
  }

  std::vector<SymbolLocation> points() {
    auto &SM = AST.sourceManager();
    auto FID = SM.getMainFileID();
    auto Offsets = Target.points();
    std::vector<SymbolLocation> Results;
    for (auto &Offset : Offsets)
      Results.emplace_back(SM.getComposedLoc(FID, Offset));
    return Results;
  }
};

TEST(LocateSymbol, Decl) {
  // Looks for decl with name 'foo' and performs locateSymbol on it.
  // Expects all the locations in the case to be returned as a location.
  const llvm::StringLiteral Cases[] = {
      "struct ^foo; struct ^foo {};",
      "namespace ns { void ^foo(); void ^foo() {} }",
      "enum class ^foo; enum class ^foo {};",
  };

  for (auto &Case : Cases) {
    SCOPED_TRACE(Case);
    LocateExample Test(Case);
    EXPECT_THAT(locateSymbol(Test.findDecl("foo")),
                ElementsAreArray(Test.points()));
  }
}

TEST(LocateSymbol, Stdlib) {
  LocateExample Test("namespace std { struct vector; }");
  EXPECT_THAT(locateSymbol(Test.findDecl("vector")),
              ElementsAre(*tooling::stdlib::Symbol::named("std::", "vector")));
}

TEST(LocateSymbol, Macros) {
  // Make sure we preserve the last one.
  LocateExample Test("#define FOO\n#undef FOO\n#define ^FOO");
  EXPECT_THAT(locateSymbol(Test.findMacro("FOO")),
              ElementsAreArray(Test.points()));
}

} // namespace
} // namespace clang::include_cleaner

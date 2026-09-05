//===- unittests/Sema/APINotesSelectorTest.cpp ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Sema/APINotesSelector.h"
#include "clang/APINotes/Types.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "gtest/gtest.h"
#include <optional>
#include <string>
#include <vector>

using namespace clang;

namespace {

using clang::ast_matchers::functionDecl;
using clang::ast_matchers::hasName;
using clang::ast_matchers::match;
using clang::ast_matchers::unless;
using clang::tooling::buildASTFromCodeWithArgs;

llvm::SmallVector<std::string, 4>
makeParameterList(llvm::ArrayRef<llvm::StringRef> Parameters) {
  llvm::SmallVector<std::string, 4> Result;
  for (llvm::StringRef Parameter : Parameters)
    Result.push_back(Parameter.str());
  return Result;
}

std::string formatSelector(llvm::ArrayRef<std::string> Parameters) {
  return api_notes::formatAPINotesParameterSelector(Parameters);
}

void expectParameterList(llvm::ArrayRef<std::string> Actual,
                         llvm::ArrayRef<llvm::StringRef> ExpectedRefs,
                         llvm::StringRef Label) {
  llvm::SmallVector<std::string, 4> Expected = makeParameterList(ExpectedRefs);

  EXPECT_EQ(Actual.size(), Expected.size())
      << Label << " selector: expected " << formatSelector(Expected) << ", got "
      << formatSelector(Actual);
  if (Actual.size() != Expected.size())
    return;

  for (unsigned I = 0, E = Expected.size(); I != E; ++I) {
    EXPECT_EQ(Actual[I], Expected[I])
        << Label << " selector: expected " << formatSelector(Expected)
        << ", got " << formatSelector(Actual);
  }
}

const FunctionDecl *findTarget(ASTUnit &AST) {
  auto Results =
      match(functionDecl(hasName("target"), unless(ast_matchers::isImplicit()))
                .bind("fn"),
            AST.getASTContext());
  EXPECT_EQ(Results.size(), 1u);
  if (Results.size() != 1u)
    return nullptr;
  return Results[0].getNodeAs<FunctionDecl>("fn");
}

void expectSelectorsImpl(
    llvm::StringRef Code, llvm::ArrayRef<llvm::StringRef> Source,
    std::optional<llvm::ArrayRef<llvm::StringRef>> Desugared,
    bool IsObjectiveCXX) {
  std::vector<std::string> Args = {"-std=c++20"};
  std::string FileName = IsObjectiveCXX ? "input.mm" : "input.cpp";

  std::unique_ptr<ASTUnit> AST = buildASTFromCodeWithArgs(Code, Args, FileName);
  ASSERT_TRUE(AST);

  const FunctionDecl *Target = findTarget(*AST);
  ASSERT_NE(Target, nullptr);

  std::optional<APINotesParameterSelectorCandidates> Candidates =
      getAPINotesParameterSelectorCandidates(AST->getASTContext(), Target);
  ASSERT_TRUE(Candidates);

  expectParameterList(Candidates->Source, Source, "source");

  EXPECT_EQ(Candidates->Desugared.has_value(), Desugared.has_value());
  if (Desugared && Candidates->Desugared)
    expectParameterList(*Candidates->Desugared, *Desugared, "desugared");
}

void expectSelectors(llvm::StringRef Code,
                     llvm::ArrayRef<llvm::StringRef> Source,
                     bool IsObjectiveCXX = false) {
  expectSelectorsImpl(Code, Source, std::nullopt, IsObjectiveCXX);
}

void expectSelectorsWithDesugared(llvm::StringRef Code,
                                  llvm::ArrayRef<llvm::StringRef> Source,
                                  llvm::ArrayRef<llvm::StringRef> Desugared) {
  expectSelectorsImpl(Code, Source, Desugared, /*IsObjectiveCXX=*/false);
}

TEST(APINotesSelectorTest, ExtractsZeroParameterSelector) {
  expectSelectors("void target();", {});
}

TEST(APINotesSelectorTest, ExtractsMultipleParametersAndIgnoresDefaults) {
  expectSelectors("void target(int, double = 0);", {"int", "double"});
}

TEST(APINotesSelectorTest, DropsTopLevelConstFromValueParameter) {
  expectSelectors("void target(const int);", {"int"});
}

TEST(APINotesSelectorTest, NormalizesPointerAndReferenceSpacing) {
  expectSelectors("void target(int *, int &, int &&);",
                  {"int*", "int&", "int&&"});
}

TEST(APINotesSelectorTest, DropsTopLevelConstFromPointerValueParameter) {
  expectSelectors("void target(int *const);", {"int*"});
}

TEST(APINotesSelectorTest, PreservesPointeeConstOnPointerParameter) {
  expectSelectors("void target(const int *);", {"const int*"});
}

TEST(APINotesSelectorTest, PreservesNestedPointerConst) {
  expectSelectors("void target(const char *const *);", {"const char*const*"});
}

TEST(APINotesSelectorTest, PreservesMemberFunctionPointerConst) {
  expectSelectors(R"cpp(
    struct Foo {};
    void target(void (Foo::*)() const);
  )cpp",
                  {"void (Foo::*)() const"});
}

TEST(APINotesSelectorTest, NormalizesParameterSelectorSpellingsDirectly) {
  EXPECT_EQ(api_notes::normalizeAPINotesParameterSelector("  unsigned   int  "),
            "unsigned int");
  EXPECT_EQ(api_notes::normalizeAPINotesParameterSelector("unsigned"),
            "unsigned int");
  EXPECT_EQ(api_notes::normalizeAPINotesParameterSelector("const unsigned"),
            "unsigned int");
  EXPECT_EQ(api_notes::normalizeAPINotesParameterSelector("unsigned const"),
            "unsigned int");
  EXPECT_EQ(api_notes::normalizeAPINotesParameterSelector("int * const"),
            "int*");
  EXPECT_EQ(
      api_notes::normalizeAPINotesParameterSelector("void (Foo::*)() const"),
      "void (Foo::*)() const");
}

TEST(APINotesSelectorTest, NormalizesTemplateSpacing) {
  expectSelectors(R"cpp(
    template <typename T, typename U> struct Box {};
    void target(Box<int, double>);
  )cpp",
                  {"Box<int,double>"});
}

TEST(APINotesSelectorTest,
     PreservesAliasAsSourceSelectorWithDesugaredFallback) {
  expectSelectorsWithDesugared(R"cpp(
    using AliasInt = int;
    void target(AliasInt);
  )cpp",
                               {"AliasInt"}, {"int"});
}

TEST(APINotesSelectorTest,
     PreservesDeepAliasAsSourceSelectorWithDesugaredFallback) {
  expectSelectorsWithDesugared(R"cpp(
    using AliasInt = int;
    using DeepAliasInt = AliasInt;
    void target(DeepAliasInt);
  )cpp",
                               {"DeepAliasInt"}, {"int"});
}

TEST(APINotesSelectorTest, StripsParameterNullability) {
  expectSelectors("void target(char * _Nonnull);", {"char*"},
                  /*IsObjectiveCXX=*/true);
}

} // namespace

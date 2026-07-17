//===- CppBoundedBuffersTest.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/ScalableStaticAnalysis/SourceTransformation/Transformations/CppBoundedBuffers.h"
#include "FindDecl.h"
#include "TestFixture.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/Basic/Sarif.h"
#include "clang/ScalableStaticAnalysis/Analyses/EntityPointerLevel/EntityPointerLevel.h"
#include "clang/ScalableStaticAnalysis/Analyses/UnsafeBufferUsage/UnsafeBufferUsageAnalysis.h"
#include "clang/ScalableStaticAnalysis/Core/ASTEntityMapping.h"
#include "clang/ScalableStaticAnalysis/Core/Model/EntityId.h"
#include "clang/ScalableStaticAnalysis/Core/Model/EntityIdTable.h"
#include "clang/ScalableStaticAnalysis/Core/Model/EntityName.h"
#include "clang/ScalableStaticAnalysis/Core/WholeProgramAnalysis/WPASuite.h"
#include "clang/ScalableStaticAnalysis/SourceTransformation/SourceEditEmitter.h"
#include "clang/ScalableStaticAnalysis/SourceTransformation/TransformationReportEmitter.h"
#include "clang/Tooling/Core/Replacement.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Error.h"
#include "gtest/gtest.h"
#include <memory>
#include <optional>
#include <string>
#include <vector>

using namespace clang;
using namespace clang::ssaf;

namespace {

class RecordingEditEmitter : public SourceEditEmitter {
public:
  std::vector<tooling::Replacement> Replacements;

  void addReplacement(tooling::Replacement R) override {
    Replacements.push_back(std::move(R));
  }
};

class RecordingReportEmitter : public TransformationReportEmitter {
public:
  struct Entry {
    std::string RuleId;
    SarifResultLevel Level;
    std::string Message;
  };
  std::vector<Entry> Results;

  void addResult(StringRef RuleId, SarifResultLevel Level, CharSourceRange,
                 StringRef Message) override {
    Results.push_back({RuleId.str(), Level, Message.str()});
  }
};

std::optional<EntityName> varEntity(StringRef Name, ASTContext &Ctx) {
  return getEntityName(findDeclByName<VarDecl>(Name, Ctx));
}

std::optional<EntityName> fieldEntity(StringRef Name, ASTContext &Ctx) {
  return getEntityName(findDeclByName<FieldDecl>(Name, Ctx));
}

std::optional<EntityName> paramEntity(StringRef Fn, unsigned Idx,
                                      ASTContext &Ctx) {
  const FunctionDecl *FD = findFnByName(Fn, Ctx);
  return FD ? getEntityName(FD->getParamDecl(Idx)) : std::nullopt;
}

std::optional<EntityName> returnEntity(StringRef Fn, ASTContext &Ctx) {
  return getEntityNameForReturn(findFnByName(Fn, Ctx));
}

struct Captured {
  std::string Rewritten;
  std::vector<RecordingReportEmitter::Entry> Reports;
};

class CppBoundedBuffersTest : public TestFixture {
protected:
  using EntityFn = llvm::function_ref<std::optional<EntityName>(ASTContext &)>;
  using MarkFn = llvm::function_ref<void(
      ASTContext &, WPASuite &, UnsafeBufferReachableAnalysisResult &)>;

  // Marks the entity \p Name reachable at \p Levels in \p Result.
  static void markReachable(WPASuite &Suite,
                            UnsafeBufferReachableAnalysisResult &Result,
                            std::optional<EntityName> Name,
                            ArrayRef<unsigned> Levels) {
    if (!Name || Levels.empty())
      return;
    EntityId Id = getIdTable(Suite).getId(*Name);
    EntityPointerLevelSet Set;
    for (unsigned Level : Levels)
      Set.insert(buildEntityPointerLevel(Id, Level));
    Result.Reachables[Id] = std::move(Set);
  }

  // Parses \p Code, lets \p Mark populate the reachable result, runs the
  // transformation, and returns the rewritten source and report entries.
  Captured runMarked(StringRef Code, MarkFn Mark) {
    std::unique_ptr<ASTUnit> AST = tooling::buildASTFromCode(Code);
    ASTContext &Ctx = AST->getASTContext();

    WPASuite Suite = makeWPASuite();
    auto Result = std::make_unique<UnsafeBufferReachableAnalysisResult>();
    Mark(Ctx, Suite, *Result);
    getData(Suite)[UnsafeBufferReachableAnalysisResult::analysisName()] =
        std::move(Result);

    RecordingEditEmitter Edits;
    RecordingReportEmitter Report;
    CppBoundedBuffers(Suite, Edits, Report).HandleTranslationUnit(Ctx);

    tooling::Replacements Replacements;
    for (const tooling::Replacement &R : Edits.Replacements)
      cantFail(Replacements.add(R));
    return {cantFail(tooling::applyAllReplacements(Code, Replacements)),
            std::move(Report.Results)};
  }

  Captured run(StringRef Code, EntityFn EntityOf, ArrayRef<unsigned> Levels) {
    return runMarked(Code, [&](ASTContext &Ctx, WPASuite &Suite,
                               UnsafeBufferReachableAnalysisResult &Result) {
      markReachable(Suite, Result, EntityOf(Ctx), Levels);
    });
  }
};

//===----------------------------------------------------------------------===//
// Rewrites: assert the rewritten source and that nothing is reported.
//===----------------------------------------------------------------------===//

TEST_F(CppBoundedBuffersTest, PointerLocal) {
  Captured C = run("void f() { int *p; }",
                   [](ASTContext &Ctx) { return varEntity("p", Ctx); }, {1});
  EXPECT_EQ(C.Rewritten, "void f() { bounded_ptr<int> p; }");
  EXPECT_TRUE(C.Reports.empty());
}

TEST_F(CppBoundedBuffersTest, PointerParameter) {
  Captured C =
      run("void f(int *p);",
          [](ASTContext &Ctx) { return paramEntity("f", 0, Ctx); }, {1});
  EXPECT_EQ(C.Rewritten, "void f(bounded_ptr<int> p);");
  EXPECT_TRUE(C.Reports.empty());
}

TEST_F(CppBoundedBuffersTest, ConstQualifiedPointee) {
  Captured C = run("const char *s;",
                   [](ASTContext &Ctx) { return varEntity("s", Ctx); }, {1});
  EXPECT_EQ(C.Rewritten, "bounded_ptr<const char> s;");
  EXPECT_TRUE(C.Reports.empty());
}

TEST_F(CppBoundedBuffersTest, VoidPointer) {
  Captured C =
      run("void *p;", [](ASTContext &Ctx) { return varEntity("p", Ctx); }, {1});
  EXPECT_EQ(C.Rewritten, "bounded_ptr<char> p;");
  EXPECT_TRUE(C.Reports.empty());
}

TEST_F(CppBoundedBuffersTest, ArrayField) {
  Captured C = run("struct S { int a[10]; };",
                   [](ASTContext &Ctx) { return fieldEntity("a", Ctx); }, {1});
  EXPECT_EQ(C.Rewritten, "struct S { bounded_array<int, 10> a; };");
  EXPECT_TRUE(C.Reports.empty());
}

TEST_F(CppBoundedBuffersTest, FunctionReturn) {
  Captured C =
      run("int *foo();",
          [](ASTContext &Ctx) { return returnEntity("foo", Ctx); }, {1});
  EXPECT_EQ(C.Rewritten, "bounded_ptr<int> foo();");
  EXPECT_TRUE(C.Reports.empty());
}

TEST_F(CppBoundedBuffersTest, GlobalPointer) {
  Captured C =
      run("int *g;", [](ASTContext &Ctx) { return varEntity("g", Ctx); }, {1});
  EXPECT_EQ(C.Rewritten, "bounded_ptr<int> g;");
  EXPECT_TRUE(C.Reports.empty());
}

TEST_F(CppBoundedBuffersTest, PointerField) {
  Captured C = run("struct S { int *p; };",
                   [](ASTContext &Ctx) { return fieldEntity("p", Ctx); }, {1});
  EXPECT_EQ(C.Rewritten, "struct S { bounded_ptr<int> p; };");
  EXPECT_TRUE(C.Reports.empty());
}

TEST_F(CppBoundedBuffersTest, ArrayOfPointers) {
  Captured C = run("int *a[10];",
                   [](ASTContext &Ctx) { return varEntity("a", Ctx); }, {1});
  EXPECT_EQ(C.Rewritten, "bounded_array<int *, 10> a;");
  EXPECT_TRUE(C.Reports.empty());
}

TEST_F(CppBoundedBuffersTest, ArrayOfFunctionPointers) {
  // A function-pointer element is accepted (not rejected like a bare function
  // pointer); the typedef keeps the declarator a clean prefix + [N] suffix.
  Captured C = run("typedef void (*FP)(); FP fps[4];",
                   [](ASTContext &Ctx) { return varEntity("fps", Ctx); }, {1});
  EXPECT_EQ(C.Rewritten, "typedef void (*FP)(); bounded_array<FP, 4> fps;");
  EXPECT_TRUE(C.Reports.empty());
}

//===----------------------------------------------------------------------===//
// Skips: assert no edit and a single reported reason.
//===----------------------------------------------------------------------===//

void expectSkip(const Captured &C, StringRef Original, ReportReason Reason) {
  EXPECT_EQ(C.Rewritten, Original);
  ASSERT_EQ(C.Reports.size(), 1u);
  EXPECT_EQ(C.Reports[0].Level, SarifResultLevel::Note);
  EXPECT_EQ(C.Reports[0].Message, messageFor(Reason).str());
}

TEST_F(CppBoundedBuffersTest, MultiLevelPointer) {
  StringRef Code = "int **pp;";
  Captured C =
      run(Code, [](ASTContext &Ctx) { return varEntity("pp", Ctx); }, {1});
  expectSkip(C, Code, ReportReason::MultiLevelPointer);
}

TEST_F(CppBoundedBuffersTest, MultiDimensionalArray) {
  StringRef Code = "int a[2][3];";
  Captured C =
      run(Code, [](ASTContext &Ctx) { return varEntity("a", Ctx); }, {1});
  expectSkip(C, Code, ReportReason::MultiDimensionalArray);
}

TEST_F(CppBoundedBuffersTest, IncompleteArray) {
  StringRef Code = "extern int a[];";
  Captured C =
      run(Code, [](ASTContext &Ctx) { return varEntity("a", Ctx); }, {1});
  expectSkip(C, Code, ReportReason::IncompleteArray);
}

TEST_F(CppBoundedBuffersTest, PointerToArray) {
  StringRef Code = "int (*p)[10];";
  Captured C =
      run(Code, [](ASTContext &Ctx) { return varEntity("p", Ctx); }, {1});
  expectSkip(C, Code, ReportReason::PointerToArray);
}

TEST_F(CppBoundedBuffersTest, ReferenceToPointer) {
  StringRef Code = "void f(int *&r);";
  Captured C =
      run(Code, [](ASTContext &Ctx) { return paramEntity("f", 0, Ctx); }, {1});
  expectSkip(C, Code, ReportReason::ReferenceToPointer);
}

TEST_F(CppBoundedBuffersTest, UnreproducibleType) {
  StringRef Code = "struct { int x; } *p;";
  Captured C =
      run(Code, [](ASTContext &Ctx) { return varEntity("p", Ctx); }, {1});
  expectSkip(C, Code, ReportReason::UnreproducibleType);
}

TEST_F(CppBoundedBuffersTest, DeclarationGroup) {
  // Both declarators share one type specifier; comma-group splitting is not
  // yet supported, so both are reported and neither is rewritten.
  Captured C =
      runMarked("int *p, *q;", [](ASTContext &Ctx, WPASuite &Suite,
                                  UnsafeBufferReachableAnalysisResult &Result) {
        markReachable(Suite, Result, varEntity("p", Ctx), {1});
        markReachable(Suite, Result, varEntity("q", Ctx), {1});
      });
  EXPECT_EQ(C.Rewritten, "int *p, *q;");
  ASSERT_EQ(C.Reports.size(), 2u);
  for (const auto &R : C.Reports) {
    EXPECT_EQ(R.Level, SarifResultLevel::Note);
    EXPECT_EQ(R.Message, messageFor(ReportReason::DeclarationGroup).str());
  }
}

TEST_F(CppBoundedBuffersTest, MacroSpelledDeclarator) {
  StringRef Code = "#define PTR int *\nPTR p;\n";
  Captured C =
      run(Code, [](ASTContext &Ctx) { return varEntity("p", Ctx); }, {1});
  expectSkip(C, Code, ReportReason::MacroExpansion);
}

TEST_F(CppBoundedBuffersTest, TrailingReturnType) {
  StringRef Code = "auto f() -> int *;";
  Captured C =
      run(Code, [](ASTContext &Ctx) { return returnEntity("f", Ctx); }, {1});
  expectSkip(C, Code, ReportReason::TrailingReturnType);
}

TEST_F(CppBoundedBuffersTest, EmissionFailureOnRawFunctionPointerArray) {
  // A raw array-of-function-pointers has no clean prefix + [N] suffix, so the
  // edit cannot be formed and the entity is reported rather than mangled.
  StringRef Code = "void (*fps[4])();";
  Captured C =
      run(Code, [](ASTContext &Ctx) { return varEntity("fps", Ctx); }, {1});
  expectSkip(C, Code, ReportReason::EmissionFailed);
}

//===----------------------------------------------------------------------===//
// Completeness and negative cases.
//===----------------------------------------------------------------------===//

TEST_F(CppBoundedBuffersTest, UnaccountedReachableIsReported) {
  // A single pointer has only level 1; marking level 2 reachable leaves the
  // entity neither rewritten nor shape-skipped, so the sweep reports it.
  StringRef Code = "int *p;";
  Captured C =
      run(Code, [](ASTContext &Ctx) { return varEntity("p", Ctx); }, {2});
  expectSkip(C, Code, ReportReason::NotTransformed);
}

TEST_F(CppBoundedBuffersTest, NotReachable) {
  StringRef Code = "int *p;";
  Captured C =
      run(Code, [](ASTContext &Ctx) { return varEntity("p", Ctx); }, {});
  EXPECT_EQ(C.Rewritten, Code);
  EXPECT_TRUE(C.Reports.empty());
}

TEST_F(CppBoundedBuffersTest, RewriteAndReportCoexist) {
  // A rewritten entity and a reported one in the same TU: the rewrite happens
  // and only the not-rewritten entity is reported.
  Captured C = runMarked(
      "int *good; int **bad;", [](ASTContext &Ctx, WPASuite &Suite,
                                  UnsafeBufferReachableAnalysisResult &Result) {
        markReachable(Suite, Result, varEntity("good", Ctx), {1});
        markReachable(Suite, Result, varEntity("bad", Ctx), {1});
      });
  EXPECT_EQ(C.Rewritten, "bounded_ptr<int> good; int **bad;");
  ASSERT_EQ(C.Reports.size(), 1u);
  EXPECT_EQ(C.Reports[0].Message,
            messageFor(ReportReason::MultiLevelPointer).str());
}

//===----------------------------------------------------------------------===//
// Classifier unit tests: pin the level direction and message coverage.
//===----------------------------------------------------------------------===//

QualType typeOf(StringRef Name, ASTContext &Ctx) {
  return findDeclByName<VarDecl>(Name, Ctx)->getType();
}

TEST_F(CppBoundedBuffersTest, ClassifyRewritesOutermostReachablePointer) {
  auto AST = tooling::buildASTFromCode("int *p;");
  llvm::SmallSet<unsigned, 4> Levels;
  Levels.insert(1);
  ClassifyResult R = classifyDeclType(typeOf("p", AST->getASTContext()), Levels,
                                      AST->getASTContext());
  ASSERT_TRUE(R.NewType.has_value());
  EXPECT_EQ(*R.NewType, BoundedType::Ptr);
  EXPECT_EQ(R.InnerSpelling, "int");
  EXPECT_FALSE(R.Skip.has_value());
}

TEST_F(CppBoundedBuffersTest, ClassifyIgnoresInnerOnlyReachablePointer) {
  auto AST = tooling::buildASTFromCode("int *p;");
  llvm::SmallSet<unsigned, 4> Levels;
  Levels.insert(2);
  ClassifyResult R = classifyDeclType(typeOf("p", AST->getASTContext()), Levels,
                                      AST->getASTContext());
  EXPECT_FALSE(R.NewType.has_value());
  EXPECT_FALSE(R.Skip.has_value());
}

TEST_F(CppBoundedBuffersTest, ClassifyMultiLevelPointerIsSkipped) {
  auto AST = tooling::buildASTFromCode("int **pp;");
  llvm::SmallSet<unsigned, 4> Levels;
  Levels.insert(1);
  ClassifyResult R = classifyDeclType(typeOf("pp", AST->getASTContext()),
                                      Levels, AST->getASTContext());
  EXPECT_FALSE(R.NewType.has_value());
  ASSERT_TRUE(R.Skip.has_value());
  EXPECT_EQ(*R.Skip, ReportReason::MultiLevelPointer);
}

TEST_F(CppBoundedBuffersTest, MessageForIsNonEmpty) {
  for (ReportReason Reason :
       {ReportReason::MultiLevelPointer, ReportReason::PointerToArray,
        ReportReason::ReferenceToPointer, ReportReason::MultiDimensionalArray,
        ReportReason::IncompleteArray, ReportReason::UnreproducibleType,
        ReportReason::DeclarationGroup, ReportReason::MacroExpansion,
        ReportReason::TrailingReturnType, ReportReason::EmissionFailed,
        ReportReason::NotTransformed})
    EXPECT_FALSE(messageFor(Reason).empty());
}

} // namespace

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
#include "clang/Frontend/SSAFOptions.h"
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
#include "llvm/ADT/StringSet.h"
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
    std::unique_ptr<ASTUnit> AST =
        tooling::buildASTFromCodeWithArgs(Code, {"-std=c++20"});
    ASTContext &Ctx = AST->getASTContext();

    WPASuite Suite = makeWPASuite();
    auto Result = std::make_unique<UnsafeBufferReachableAnalysisResult>();
    Mark(Ctx, Suite, *Result);
    getData(Suite)[UnsafeBufferReachableAnalysisResult::analysisName()] =
        std::move(Result);

    RecordingEditEmitter Edits;
    RecordingReportEmitter Report;
    SSAFOptions Opts;
    CppBoundedBuffers(Suite, Opts, Edits, Report).HandleTranslationUnit(Ctx);

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
  ASSERT_TRUE(C.Rewritten == "void f() { bounded_ptr<int> p; }");
  EXPECT_TRUE(C.Reports.empty());
}

TEST_F(CppBoundedBuffersTest, PointerParameter) {
  Captured C =
      run("void f(int *p);",
          [](ASTContext &Ctx) { return paramEntity("f", 0, Ctx); }, {1});
  ASSERT_TRUE(C.Rewritten == "void f(bounded_ptr<int> p);");
  EXPECT_TRUE(C.Reports.empty());
}

TEST_F(CppBoundedBuffersTest, ConstQualifiedPointee) {
  Captured C = run("const char *s;",
                   [](ASTContext &Ctx) { return varEntity("s", Ctx); }, {1});
  ASSERT_TRUE(C.Rewritten == "bounded_ptr<const char> s;");
  EXPECT_TRUE(C.Reports.empty());
}

TEST_F(CppBoundedBuffersTest, VoidPointer) {
  Captured C =
      run("void *p;", [](ASTContext &Ctx) { return varEntity("p", Ctx); }, {1});
  ASSERT_TRUE(C.Rewritten == "bounded_ptr<char> p;");
  EXPECT_TRUE(C.Reports.empty());
}

TEST_F(CppBoundedBuffersTest, ArrayField) {
  Captured C = run("struct S { int a[10]; };",
                   [](ASTContext &Ctx) { return fieldEntity("a", Ctx); }, {1});
  ASSERT_TRUE(C.Rewritten == "struct S { bounded_array<int, 10> a; };");
  EXPECT_TRUE(C.Reports.empty());
}

TEST_F(CppBoundedBuffersTest, FunctionReturn) {
  Captured C =
      run("int *foo();",
          [](ASTContext &Ctx) { return returnEntity("foo", Ctx); }, {1});
  ASSERT_TRUE(C.Rewritten == "bounded_ptr<int> foo();");
  EXPECT_TRUE(C.Reports.empty());
}

TEST_F(CppBoundedBuffersTest, GlobalPointer) {
  Captured C =
      run("int *g;", [](ASTContext &Ctx) { return varEntity("g", Ctx); }, {1});
  ASSERT_TRUE(C.Rewritten == "bounded_ptr<int> g;");
  EXPECT_TRUE(C.Reports.empty());
}

TEST_F(CppBoundedBuffersTest, PointerField) {
  Captured C = run("struct S { int *p; };",
                   [](ASTContext &Ctx) { return fieldEntity("p", Ctx); }, {1});
  ASSERT_TRUE(C.Rewritten == "struct S { bounded_ptr<int> p; };");
  EXPECT_TRUE(C.Reports.empty());
}

TEST_F(CppBoundedBuffersTest, ArrayOfPointers) {
  Captured C = run("int *a[10];",
                   [](ASTContext &Ctx) { return varEntity("a", Ctx); }, {1});
  ASSERT_TRUE(C.Rewritten == "bounded_array<int *, 10>a;");
  EXPECT_TRUE(C.Reports.empty());
}

TEST_F(CppBoundedBuffersTest, ArrayOfFunctionPointers) {
  // A function-pointer element is accepted (not rejected like a bare function
  // pointer); the typedef keeps the declarator a clean prefix + [N] suffix.
  Captured C = run("typedef void (*FP)(); FP fps[4];",
                   [](ASTContext &Ctx) { return varEntity("fps", Ctx); }, {1});
  ASSERT_TRUE(C.Rewritten == "typedef void (*FP)(); bounded_array<FP, 4> fps;");
  EXPECT_TRUE(C.Reports.empty());
}

TEST_F(CppBoundedBuffersTest, ConstQualifiedPointeeSpelledAfter) {
  // `char const *` means the same as `const char *`; the qualifier belongs to
  // the pointee either way and is reproduced inside the angle brackets.
  Captured C = run("char const *s;",
                   [](ASTContext &Ctx) { return varEntity("s", Ctx); }, {1});
  ASSERT_TRUE(C.Rewritten == "bounded_ptr<const char> s;");
  EXPECT_TRUE(C.Reports.empty());
}

TEST_F(CppBoundedBuffersTest, ConstVolatileQualifiedPointee) {
  Captured C = run("const volatile char *s;",
                   [](ASTContext &Ctx) { return varEntity("s", Ctx); }, {1});
  ASSERT_TRUE(C.Rewritten == "bounded_ptr<const volatile char> s;");
  EXPECT_TRUE(C.Reports.empty());
}

TEST_F(CppBoundedBuffersTest, ConstPointerKeepsItsOwnQualifier) {
  // The `const` applies to the pointer, not the pointee, so it lies outside the
  // rewrite range and stays where it was written.
  Captured C = run("int *const p = nullptr;",
                   [](ASTContext &Ctx) { return varEntity("p", Ctx); }, {1});
  ASSERT_TRUE(C.Rewritten == "bounded_ptr<int> const p = nullptr;");
  EXPECT_TRUE(C.Reports.empty());
}

TEST_F(CppBoundedBuffersTest, ConstPointerToConstPointee) {
  Captured C = run("const int *const p = nullptr;",
                   [](ASTContext &Ctx) { return varEntity("p", Ctx); }, {1});
  ASSERT_TRUE(C.Rewritten == "bounded_ptr<const int> const p = nullptr;");
  EXPECT_TRUE(C.Reports.empty());
}

TEST_F(CppBoundedBuffersTest, StorageClassBeforeQualifiedPointee) {
  // `static` precedes the qualifier run and is left untouched.
  Captured C = run("static const char *s;",
                   [](ASTContext &Ctx) { return varEntity("s", Ctx); }, {1});
  ASSERT_TRUE(C.Rewritten == "static bounded_ptr<const char> s;");
  EXPECT_TRUE(C.Reports.empty());
}

TEST_F(CppBoundedBuffersTest, ConstQualifiedArrayElement) {
  Captured C = run("const int a[10] = {};",
                   [](ASTContext &Ctx) { return varEntity("a", Ctx); }, {1});
  ASSERT_TRUE(C.Rewritten == "bounded_array<const int, 10> a = {};");
  EXPECT_TRUE(C.Reports.empty());
}

TEST_F(CppBoundedBuffersTest, ConstQualifiedArrayElementSpelledAfter) {
  Captured C = run("int const a[10] = {};",
                   [](ASTContext &Ctx) { return varEntity("a", Ctx); }, {1});
  ASSERT_TRUE(C.Rewritten == "bounded_array<const int, 10> a = {};");
  EXPECT_TRUE(C.Reports.empty());
}

TEST_F(CppBoundedBuffersTest, QualifiedPointerFunctionReturn) {
  Captured C =
      run("const char *foo();",
          [](ASTContext &Ctx) { return returnEntity("foo", Ctx); }, {1});
  ASSERT_TRUE(C.Rewritten == "bounded_ptr<const char> foo();");
  EXPECT_TRUE(C.Reports.empty());
}

TEST_F(CppBoundedBuffersTest, MultipleTrailingPointerQualifiers) {
  Captured C = run("int *volatile const p = nullptr;",
                   [](ASTContext &Ctx) { return varEntity("p", Ctx); }, {1});
  ASSERT_TRUE(C.Rewritten == "bounded_ptr<int> volatile const p = nullptr;");
  EXPECT_TRUE(C.Reports.empty());
}

TEST_F(CppBoundedBuffersTest, LeadingAndMultipleTrailingPointerQualifiers) {
  Captured C = run("const int *const volatile p = nullptr;",
                   [](ASTContext &Ctx) { return varEntity("p", Ctx); }, {1});
  ASSERT_TRUE(C.Rewritten ==
              "bounded_ptr<const int> const volatile p = nullptr;");
  EXPECT_TRUE(C.Reports.empty());
}

TEST_F(CppBoundedBuffersTest, MultipleTrailingArrayQualifiers) {
  // Both qualify the element, so the range grows right over the whole run.
  Captured C = run("int const volatile a[10] = {};",
                   [](ASTContext &Ctx) { return varEntity("a", Ctx); }, {1});
  ASSERT_TRUE(C.Rewritten == "bounded_array<const volatile int, 10> a = {};");
  EXPECT_TRUE(C.Reports.empty());
}

TEST_F(CppBoundedBuffersTest, MultipleTrailingArrayQualifiersReversed) {
  Captured C = run("int volatile const a[10] = {};",
                   [](ASTContext &Ctx) { return varEntity("a", Ctx); }, {1});
  ASSERT_TRUE(C.Rewritten == "bounded_array<const volatile int, 10> a = {};");
  EXPECT_TRUE(C.Reports.empty());
}

TEST_F(CppBoundedBuffersTest, CommentBetweenPointeeTypeAndStar) {
  Captured C = run("int /* c */ *p;",
                   [](ASTContext &Ctx) { return varEntity("p", Ctx); }, {1});
  ASSERT_TRUE(C.Rewritten == "bounded_ptr<int> p;");
  EXPECT_TRUE(C.Reports.empty());
}

TEST_F(CppBoundedBuffersTest, AttributeBetweenPointeeTypeAndStar) {
  // A type attribute also sits inside the rewrite range. It is part of the
  // pointee type, so the pretty-printed spelling reproduces it.
  Captured C = run("int __attribute__((address_space(1))) *p;",
                   [](ASTContext &Ctx) { return varEntity("p", Ctx); }, {1});
  EXPECT_EQ(C.Rewritten,
            "bounded_ptr<__attribute__((address_space(1))) int> p;");
  EXPECT_TRUE(C.Reports.empty());
}

TEST_F(CppBoundedBuffersTest, DeclAttributeAfterArrayBrackets) {
  // A declaration attribute is not part of the type-loc, so it lies beyond the
  // deleted extent and survives untouched.
  Captured C = run("int a[10] __attribute__((aligned(16)));",
                   [](ASTContext &Ctx) { return varEntity("a", Ctx); }, {1});
  EXPECT_EQ(C.Rewritten,
            "bounded_array<int, 10> a __attribute__((aligned(16)));");
  EXPECT_TRUE(C.Reports.empty());
}

TEST_F(CppBoundedBuffersTest, DeclAttributeAfterPointerDeclarator) {
  Captured C = run("int *p __attribute__((aligned(16)));",
                   [](ASTContext &Ctx) { return varEntity("p", Ctx); }, {1});
  ASSERT_TRUE(C.Rewritten ==
              "bounded_ptr<int> p __attribute__((aligned(16)));");
  EXPECT_TRUE(C.Reports.empty());
}

TEST_F(CppBoundedBuffersTest, AliasedLambdaPointee) {
  // The closure type itself is unnamed, but the alias supplies a name that can
  // be written as the template argument, and it denotes that same closure type.
  Captured C = run("using L = decltype([](int x) { return x; });\nL *p;\n",
                   [](ASTContext &Ctx) { return varEntity("p", Ctx); }, {1});
  EXPECT_EQ(C.Rewritten, "using L = decltype([](int x) { return x; });\n"
                         "bounded_ptr<L> p;\n");
  EXPECT_TRUE(C.Reports.empty());
}

//===----------------------------------------------------------------------===//
// Skips: assert no edit and a single reported reason.
//===----------------------------------------------------------------------===//

void expectSkip(const Captured &C, StringRef Original, ReportReason Reason) {
  ASSERT_TRUE(C.Rewritten == Original);
  ASSERT_TRUE(C.Reports.size() == 1u);
  ASSERT_TRUE(C.Reports[0].Level == SarifResultLevel::Note);
  ASSERT_TRUE(C.Reports[0].Message == messageFor(Reason).str());
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

TEST_F(CppBoundedBuffersTest, UnnamableType) {
  StringRef Code = "struct { int x; } *p;";
  Captured C =
      run(Code, [](ASTContext &Ctx) { return varEntity("p", Ctx); }, {1});
  expectSkip(C, Code, ReportReason::UnnamableType);
}

TEST_F(CppBoundedBuffersTest, InlineLambdaPointee) {
  // Each lambda-expression yields a distinct closure type, so there is no name
  // to write: re-spelling the expression would denote a different type.
  StringRef Code = "decltype([](int x) { return x; }) *p;";
  Captured C =
      run(Code, [](ASTContext &Ctx) { return varEntity("p", Ctx); }, {1});
  expectSkip(C, Code, ReportReason::UnnamableType);
}

TEST_F(CppBoundedBuffersTest, UnaliasedLambdaPointee) {
  // decltype of a variable names the closure type but is not a typedef-name, so
  // the unnamed record is what the check sees.
  StringRef Code =
      "void f() { auto lam = [](int x) { return x; }; decltype(lam) *p; }";
  Captured C =
      run(Code, [](ASTContext &Ctx) { return varEntity("p", Ctx); }, {1});
  expectSkip(C, Code, ReportReason::UnnamableType);
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
  ASSERT_TRUE(C.Rewritten == "int *p, *q;");
  ASSERT_TRUE(C.Reports.size() == 2u);
  for (const auto &R : C.Reports) {
    ASSERT_TRUE(R.Level == SarifResultLevel::Note);
    ASSERT_TRUE(R.Message == messageFor(ReportReason::DeclarationGroup).str());
  }
}

TEST_F(CppBoundedBuffersTest, GlobalDeclarationGroupOfThree) {
  // A three-way (not just two-way) comma group at namespace scope; every
  // declarator, including the multi-level pointer, is reported.
  StringRef Code = "extern int *const p, *const q, *volatile *pp;";
  Captured C = runMarked(Code, [](ASTContext &Ctx, WPASuite &Suite,
                                  UnsafeBufferReachableAnalysisResult &Result) {
    markReachable(Suite, Result, varEntity("p", Ctx), {1});
    markReachable(Suite, Result, varEntity("q", Ctx), {1});
    markReachable(Suite, Result, varEntity("pp", Ctx), {1, 2});
  });
  ASSERT_TRUE(C.Rewritten == Code);
  ASSERT_TRUE(C.Reports.size() == 3u);
  for (const auto &R : C.Reports) {
    ASSERT_TRUE(R.Level == SarifResultLevel::Note);
    ASSERT_TRUE(R.Message == messageFor(ReportReason::DeclarationGroup).str());
  }
}

TEST_F(CppBoundedBuffersTest, FieldDeclarationGroupOfThree) {
  // Same comma group, but as FieldDecls inside a RecordDecl rather than
  // VarDecls inside the TranslationUnitDecl.
  StringRef Code = "struct Tup { int *const p, *const q, *volatile *pp; };";
  Captured C = runMarked(Code, [](ASTContext &Ctx, WPASuite &Suite,
                                  UnsafeBufferReachableAnalysisResult &Result) {
    markReachable(Suite, Result, fieldEntity("p", Ctx), {1});
    markReachable(Suite, Result, fieldEntity("q", Ctx), {1});
    markReachable(Suite, Result, fieldEntity("pp", Ctx), {1, 2});
  });
  ASSERT_TRUE(C.Rewritten == Code);
  ASSERT_TRUE(C.Reports.size() == 3u);
  for (const auto &R : C.Reports) {
    ASSERT_TRUE(R.Level == SarifResultLevel::Note);
    ASSERT_TRUE(R.Message == messageFor(ReportReason::DeclarationGroup).str());
  }
}

TEST_F(CppBoundedBuffersTest, ForInitDeclarationGroupOfThree) {
  // Same comma group again, but as a DeclStmt in a for-loop init-statement;
  // the lexical DeclContext is the enclosing function, not the loop itself.
  StringRef Code = "void test() {\n"
                   "  for (int *const p = {}, *const q = {}, "
                   "*volatile *pp = {}; true;) {\n"
                   "    return;\n"
                   "  }\n"
                   "}\n";
  Captured C = runMarked(Code, [](ASTContext &Ctx, WPASuite &Suite,
                                  UnsafeBufferReachableAnalysisResult &Result) {
    markReachable(Suite, Result, varEntity("p", Ctx), {1});
    markReachable(Suite, Result, varEntity("q", Ctx), {1});
    markReachable(Suite, Result, varEntity("pp", Ctx), {1, 2});
  });
  ASSERT_TRUE(C.Rewritten == Code);
  ASSERT_TRUE(C.Reports.size() == 3u);
  for (const auto &R : C.Reports) {
    ASSERT_TRUE(R.Level == SarifResultLevel::Note);
    ASSERT_TRUE(R.Message == messageFor(ReportReason::DeclarationGroup).str());
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

TEST_F(CppBoundedBuffersTest, RawFunctionPointerArrayDoesNotEndInBracket) {
  // A raw array-of-function-pointers has no clean prefix + [N] suffix: the
  // element spelling wraps the name, so the array type ends at the trailing
  // `()` rather than at its closing bracket.
  StringRef Code = "void (*fps[4])();";
  Captured C =
      run(Code, [](ASTContext &Ctx) { return varEntity("fps", Ctx); }, {1});
  expectSkip(C, Code, ReportReason::ArrayNotEndInBracket);
}

TEST_F(CppBoundedBuffersTest, ParenthesizedPointerDeclarator) {
  // The parens wrap the declarator, so the type ends at the ')' rather than at
  // the '*'. A range anchored on the type would span the name and unbalance the
  // parens, so the declarator is reported instead.
  StringRef Code = "int (*par);";
  Captured C =
      run(Code, [](ASTContext &Ctx) { return varEntity("par", Ctx); }, {1});
  expectSkip(C, Code, ReportReason::NotPointerTypeEndWithStar);
}

TEST_F(CppBoundedBuffersTest, TypedefSpelledPointer) {
  // The declarator spells no pointer of its own, so there is no pointee
  // type-loc to build a rewrite range from.
  StringRef Code = "typedef int *IP;\nIP p;\n";
  Captured C =
      run(Code, [](ASTContext &Ctx) { return varEntity("p", Ctx); }, {1});
  expectSkip(C, Code, ReportReason::NoInnerTypeLoc);
}

TEST_F(CppBoundedBuffersTest, QualifierSeparatedFromPointeeType) {
  // `const` is separated from the type by `static`, so absorbing it into the
  // rewrite range would need a non-contiguous edit.
  StringRef Code = "const static char *s;";
  Captured C =
      run(Code, [](ASTContext &Ctx) { return varEntity("s", Ctx); }, {1});
  expectSkip(C, Code, ReportReason::UnexpectedLeadingQualifier);
}

TEST_F(CppBoundedBuffersTest, CommentBetweenQualifierAndPointeeType) {
  // A comment interrupts the qualifier run; absorbing the `const` would delete
  // the comment along with it.
  StringRef Code = "const /* c */ char *s;";
  Captured C =
      run(Code, [](ASTContext &Ctx) { return varEntity("s", Ctx); }, {1});
  expectSkip(C, Code, ReportReason::UnexpectedLeadingQualifier);
}

TEST_F(CppBoundedBuffersTest, CommentBetweenElementTypeAndTrailingQualifier) {
  // The `const` qualifies the element, so its meaning moves inside the bounded
  // type and it must be absorbed by the rewrite range. The comment separates it
  // from the element type, which would need a non-contiguous edit.
  StringRef Code = "int /* c */ const a[10] = {};";
  Captured C =
      run(Code, [](ASTContext &Ctx) { return varEntity("a", Ctx); }, {1});
  expectSkip(C, Code, ReportReason::UnexpectedTrailingQualifier);
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
  ASSERT_TRUE(C.Rewritten == Code);
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
  ASSERT_TRUE(C.Rewritten == "bounded_ptr<int> good; int **bad;");
  ASSERT_TRUE(C.Reports.size() == 1u);
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
  ASSERT_FALSE(R.Skip.has_value());
  ASSERT_TRUE(R.NewType == BoundedType::Ptr);
  ASSERT_TRUE(R.InnerSpelling == "int");
}

TEST_F(CppBoundedBuffersTest, ClassifyIgnoresInnerOnlyReachablePointer) {
  // A single pointer has only level 1, so nothing is recognized and the
  // catch-all reason is reported rather than leaving the entity undecided.
  auto AST = tooling::buildASTFromCode("int *p;");
  llvm::SmallSet<unsigned, 4> Levels;
  Levels.insert(2);
  ClassifyResult R = classifyDeclType(typeOf("p", AST->getASTContext()), Levels,
                                      AST->getASTContext());
  ASSERT_TRUE(R.Skip.has_value());
  ASSERT_TRUE(*R.Skip == ReportReason::NotTransformed);
}

TEST_F(CppBoundedBuffersTest, ClassifyMultiLevelPointerIsSkipped) {
  auto AST = tooling::buildASTFromCode("int **pp;");
  llvm::SmallSet<unsigned, 4> Levels;
  Levels.insert(1);
  ClassifyResult R = classifyDeclType(typeOf("pp", AST->getASTContext()),
                                      Levels, AST->getASTContext());
  ASSERT_TRUE(R.Skip.has_value());
  ASSERT_TRUE(*R.Skip == ReportReason::MultiLevelPointer);
}

TEST_F(CppBoundedBuffersTest, MessageForIsNonEmpty) {
  // Walks the whole enum rather than a hand-kept list, so a reason added
  // without a message is caught here and not silently left untested.
  for (unsigned I = 0; I <= static_cast<unsigned>(ReportReason::UnnamableType);
       ++I) {
    auto Reason = static_cast<ReportReason>(I);
    EXPECT_FALSE(messageFor(Reason).empty());
  }
}

TEST_F(CppBoundedBuffersTest, MessageForIsUnique) {
  // Two reasons sharing a message would make reports ambiguous.
  llvm::StringSet<> Seen;
  for (unsigned I = 0; I <= static_cast<unsigned>(ReportReason::UnnamableType);
       ++I) {
    auto Reason = static_cast<ReportReason>(I);
    EXPECT_TRUE(Seen.insert(messageFor(Reason)).second)
        << "duplicate message: " << messageFor(Reason).str();
  }
}

} // namespace

//===- unittests/StaticAnalyzer/SvalTest.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CheckerRegistration.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclGroup.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/Type.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ExprEngine.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/MemRegion.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SVals.h"
#include "clang/StaticAnalyzer/Frontend/AnalysisConsumer.h"
#include "clang/StaticAnalyzer/Frontend/CheckerRegistry.h"
#include "clang/Testing/TestClangConfig.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

namespace clang {

// getType() tests include whole bunch of type comparisons,
// so when something is wrong, it's good to have gtest telling us
// what are those types.
[[maybe_unused]] std::ostream &operator<<(std::ostream &OS, const QualType &T) {
  return OS << T.getAsString();
}

[[maybe_unused]] std::ostream &operator<<(std::ostream &OS,
                                          const CanQualType &T) {
  return OS << QualType{T};
}

namespace ento {
namespace {

//===----------------------------------------------------------------------===//
//                       Testing framework implementation
//===----------------------------------------------------------------------===//

/// A simple map from variable names to symbolic values used to init them.
using SVals = llvm::StringMap<SVal>;

/// SValCollector is the barebone of all tests.
///
/// It is implemented as a checker and reacts to binds, so we find
/// symbolic values of interest, and to end analysis, where we actually
/// can test whatever we gathered.
class SValCollector : public Checker<check::Bind, check::EndAnalysis> {
public:
  void checkBind(SVal Loc, SVal Val, const Stmt *S, bool AtDeclInit,
                 CheckerContext &C) const {
    // Skip instantly if we finished testing.
    // Also, we care only for binds happening in variable initializations.
    if (Tested || !isa<DeclStmt>(S))
      return;

    if (const auto *VR = llvm::dyn_cast_or_null<VarRegion>(Loc.getAsRegion())) {
      CollectedSVals[VR->getDescriptiveName(false)] = Val;
    }
  }

  void checkEndAnalysis(ExplodedGraph &G, BugReporter &B,
                        ExprEngine &Engine) const {
    if (!Tested) {
      test(Engine, Engine.getContext());
      Tested = true;
      CollectedSVals.clear();
    }
  }

  /// Helper function for tests to access bound symbolic values.
  SVal getByName(StringRef Name) const { return CollectedSVals[Name]; }

private:
  /// Entry point for tests.
  virtual void test(ExprEngine &Engine, const ASTContext &Context) const = 0;

  mutable bool Tested = false;
  mutable SVals CollectedSVals;
};

static void expectSameSignAndBitWidth(QualType ExpectedTy, QualType ActualTy,
                                      const ASTContext &Context) {
  EXPECT_EQ(ExpectedTy->isUnsignedIntegerType(),
            ActualTy->isUnsignedIntegerType());
  EXPECT_EQ(Context.getTypeSize(ExpectedTy), Context.getTypeSize(ActualTy));
}

// Fixture class for parameterized SValTest
class SValTest : public testing::TestWithParam<TestClangConfig> {};

// SVAL_TEST is a combined way of providing a short code snippet and
// to test some programmatic predicates on symbolic values produced by the
// engine for the actual code.
//
// Each test has a NAME.  One can think of it as a name for normal gtests.
//
// Each test should provide a CODE snippet.  Code snippets might contain any
// valid C/C++, but have ONLY ONE defined function.  There are no requirements
// about function's name or parameters.  It can even be a class method.  The
// body of the function must contain a set of variable declarations.  Each
// variable declaration gets bound to a symbolic value, so for the following
// example:
//
//     int x = <expr>;
//
// `x` will be bound to whatever symbolic value the engine produced for <expr>.
// LIVENESS and REASSIGNMENTS don't affect this binding.
//
// During the test the actual values can be accessed via `getByName` function,
// and, for the `x`-bound value, one must use "x" as its name.
//
// Example:
// SVAL_TEST(SimpleSValTest, R"(
// void foo() {
//   int x = 42;
// })") {
//   SVal X = getByName("x");
//   EXPECT_TRUE(X.isConstant(42));
// }
#define SVAL_TEST(NAME, CODE)                                                  \
  class NAME##SValCollector final : public SValCollector {                     \
  public:                                                                      \
    void test(ExprEngine &Engine, const ASTContext &Context) const override;   \
  };                                                                           \
                                                                               \
  void add##NAME##SValCollector(AnalysisASTConsumer &AnalysisConsumer,         \
                                AnalyzerOptions &AnOpts) {                     \
    AnOpts.CheckersAndPackages = {{"test." #NAME "SValColl", true}};           \
    AnalysisConsumer.AddCheckerRegistrationFn([](CheckerRegistry &Registry) {  \
      Registry.addChecker<NAME##SValCollector>("test." #NAME "SValColl",       \
                                               "MockDescription");             \
    });                                                                        \
  }                                                                            \
                                                                               \
  TEST_P(SValTest, NAME) {                                                     \
    EXPECT_TRUE(runCheckerOnCodeWithArgs<add##NAME##SValCollector>(            \
        CODE, GetParam().getCommandLineArgs()));                               \
  }                                                                            \
  void NAME##SValCollector::test(ExprEngine &Engine,                           \
                                 const ASTContext &Context) const

//===----------------------------------------------------------------------===//
//                                 Actual tests
//===----------------------------------------------------------------------===//

SVAL_TEST(GetConstType, R"(
void foo() {
  int x = 42;
  int *y = nullptr;
  bool z = true;
})") {
  SVal X = getByName("x");
  ASSERT_FALSE(X.getType(Context).isNull());
  EXPECT_EQ(Context.IntTy, X.getType(Context));

  SVal Y = getByName("y");
  ASSERT_FALSE(Y.getType(Context).isNull());
  expectSameSignAndBitWidth(Context.getUIntPtrType(), Y.getType(Context),
                            Context);

  SVal Z = getByName("z");
  ASSERT_FALSE(Z.getType(Context).isNull());
  EXPECT_EQ(Context.BoolTy, Z.getType(Context));
}

SVAL_TEST(GetLocAsIntType, R"(
void foo(int *x) {
  long int a = (long long int)x;
  unsigned b = (long long unsigned)&a;
  int c = (long long int)nullptr;
})") {
  SVal A = getByName("a");
  ASSERT_FALSE(A.getType(Context).isNull());

  // TODO: Turn it into signed long
  expectSameSignAndBitWidth(Context.UnsignedLongTy, A.getType(Context),
                            Context);

  SVal B = getByName("b");
  ASSERT_FALSE(B.getType(Context).isNull());
  expectSameSignAndBitWidth(Context.UnsignedIntTy, B.getType(Context), Context);

  SVal C = getByName("c");
  ASSERT_FALSE(C.getType(Context).isNull());
  expectSameSignAndBitWidth(Context.IntTy, C.getType(Context), Context);
}

SVAL_TEST(GetSymExprType, R"(
void foo(int a, int b) {
  int x = a;
  int y = a + b;
  long z = a;
})") {
  QualType Int = Context.IntTy;

  SVal X = getByName("x");
  ASSERT_FALSE(X.getType(Context).isNull());
  EXPECT_EQ(Int, X.getType(Context));

  SVal Y = getByName("y");
  ASSERT_FALSE(Y.getType(Context).isNull());
  EXPECT_EQ(Int, Y.getType(Context));

  // TODO: Change to Long when we support symbolic casts
  SVal Z = getByName("z");
  ASSERT_FALSE(Z.getType(Context).isNull());
  EXPECT_EQ(Int, Z.getType(Context));
}

SVAL_TEST(GetPointerType, R"(
int *bar();
int &foobar();
struct Z {
  int a;
  int *b;
};
void foo(int x, int *y, Z z) {
  int &a = x;
  int &b = *y;
  int &c = *bar();
  int &d = foobar();
  int &e = z.a;
  int &f = *z.b;
})") {
  QualType Int = Context.IntTy;

  SVal A = getByName("a");
  ASSERT_FALSE(A.getType(Context).isNull());
  const auto *APtrTy = dyn_cast<PointerType>(A.getType(Context));
  ASSERT_NE(APtrTy, nullptr);
  EXPECT_EQ(Int, APtrTy->getPointeeType());

  SVal B = getByName("b");
  ASSERT_FALSE(B.getType(Context).isNull());
  const auto *BPtrTy = dyn_cast<PointerType>(B.getType(Context));
  ASSERT_NE(BPtrTy, nullptr);
  EXPECT_EQ(Int, BPtrTy->getPointeeType());

  SVal C = getByName("c");
  ASSERT_FALSE(C.getType(Context).isNull());
  const auto *CPtrTy = dyn_cast<PointerType>(C.getType(Context));
  ASSERT_NE(CPtrTy, nullptr);
  EXPECT_EQ(Int, CPtrTy->getPointeeType());

  SVal D = getByName("d");
  ASSERT_FALSE(D.getType(Context).isNull());
  const auto *DRefTy = dyn_cast<LValueReferenceType>(D.getType(Context));
  ASSERT_NE(DRefTy, nullptr);
  EXPECT_EQ(Int, DRefTy->getPointeeType());

  SVal E = getByName("e");
  ASSERT_FALSE(E.getType(Context).isNull());
  const auto *EPtrTy = dyn_cast<PointerType>(E.getType(Context));
  ASSERT_NE(EPtrTy, nullptr);
  EXPECT_EQ(Int, EPtrTy->getPointeeType());

  SVal F = getByName("f");
  ASSERT_FALSE(F.getType(Context).isNull());
  const auto *FPtrTy = dyn_cast<PointerType>(F.getType(Context));
  ASSERT_NE(FPtrTy, nullptr);
  EXPECT_EQ(Int, FPtrTy->getPointeeType());
}

SVAL_TEST(GetCompoundType, R"(
struct TestStruct {
  int a, b;
};
union TestUnion {
  int a;
  float b;
  TestStruct c;
};
void foo(int x) {
  int a[] = {1, x, 2};
  TestStruct b = {x, 42};
  TestUnion c = {42};
  TestUnion d = {.c=b};
}
)") {
  SVal A = getByName("a");
  ASSERT_FALSE(A.getType(Context).isNull());
  const auto *AArrayType = dyn_cast<ArrayType>(A.getType(Context));
  ASSERT_NE(AArrayType, nullptr);
  EXPECT_EQ(Context.IntTy, AArrayType->getElementType());

  SVal B = getByName("b");
  ASSERT_FALSE(B.getType(Context).isNull());
  const auto *BRecordType = dyn_cast<RecordType>(B.getType(Context));
  ASSERT_NE(BRecordType, nullptr);
  EXPECT_EQ("TestStruct", BRecordType->getDecl()->getName());

  SVal C = getByName("c");
  ASSERT_FALSE(C.getType(Context).isNull());
  const auto *CRecordType = dyn_cast<RecordType>(C.getType(Context));
  ASSERT_NE(CRecordType, nullptr);
  EXPECT_EQ("TestUnion", CRecordType->getDecl()->getName());

  auto D = getByName("d").getAs<nonloc::CompoundVal>();
  ASSERT_TRUE(D.has_value());
  auto Begin = D->begin();
  ASSERT_NE(D->end(), Begin);
  ++Begin;
  ASSERT_EQ(D->end(), Begin);
  auto LD = D->begin()->getAs<nonloc::LazyCompoundVal>();
  ASSERT_TRUE(LD.has_value());
  auto LDT = LD->getType(Context);
  ASSERT_FALSE(LDT.isNull());
  const auto *DRecordType = dyn_cast<RecordType>(LDT);
  ASSERT_NE(DRecordType, nullptr);
  EXPECT_EQ("TestStruct", DRecordType->getDecl()->getName());
}

SVAL_TEST(GetStringType, R"(
void foo() {
  const char *a = "Hello, world!";
}
)") {
  SVal A = getByName("a");
  ASSERT_FALSE(A.getType(Context).isNull());
  const auto *APtrTy = dyn_cast<PointerType>(A.getType(Context));
  ASSERT_NE(APtrTy, nullptr);
  EXPECT_EQ(Context.CharTy, APtrTy->getPointeeType());
}

SVAL_TEST(GetThisType, R"(
class TestClass {
  void foo();
};
void TestClass::foo() {
  const auto *a = this;
}
)") {
  SVal A = getByName("a");
  ASSERT_FALSE(A.getType(Context).isNull());
  const auto *APtrTy = dyn_cast<PointerType>(A.getType(Context));
  ASSERT_NE(APtrTy, nullptr);
  const auto *ARecordType = dyn_cast<RecordType>(APtrTy->getPointeeType());
  ASSERT_NE(ARecordType, nullptr);
  EXPECT_EQ("TestClass", ARecordType->getDecl()->getName());
}

SVAL_TEST(GetFunctionPtrType, R"(
void bar();
void foo() {
  auto *a = &bar;
}
)") {
  SVal A = getByName("a");
  ASSERT_FALSE(A.getType(Context).isNull());
  const auto *APtrTy = dyn_cast<PointerType>(A.getType(Context));
  ASSERT_NE(APtrTy, nullptr);
  ASSERT_TRUE(isa<FunctionProtoType>(APtrTy->getPointeeType()));
}

SVAL_TEST(GetLabelType, R"(
void foo() {
  entry:
  void *a = &&entry;
  char *b = (char *)&&entry;
}
)") {
  SVal A = getByName("a");
  ASSERT_FALSE(A.getType(Context).isNull());
  EXPECT_EQ(Context.VoidPtrTy, A.getType(Context));

  SVal B = getByName("a");
  ASSERT_FALSE(B.getType(Context).isNull());
  // TODO: Change to CharTy when we support symbolic casts
  EXPECT_EQ(Context.VoidPtrTy, B.getType(Context));
}

std::vector<TestClangConfig> allTestClangConfigs() {
  std::vector<TestClangConfig> all_configs;
  TestClangConfig config;
  config.Language = Lang_CXX14;
  for (std::string target :
       {"i686-pc-windows-msvc",   "i686-apple-darwin9",
        "x86_64-apple-darwin9",   "x86_64-scei-ps4",
        "x86_64-windows-msvc",    "x86_64-unknown-linux",
        "x86_64-apple-macosx",    "x86_64-apple-ios14.0",
        "wasm32-unknown-unknown", "wasm64-unknown-unknown",
        "thumb-pc-win32",         "sparc64-none-openbsd",
        "sparc-none-none",        "riscv64-unknown-linux",
        "ppc64-windows-msvc",     "powerpc-ibm-aix",
        "powerpc64-ibm-aix",      "s390x-ibm-zos",
        "armv7-pc-windows-msvc",  "aarch64-pc-windows-msvc",
        "xcore-xmos-elf"}) {
    config.Target = target;
    all_configs.push_back(config);
  }
  return all_configs;
}

INSTANTIATE_TEST_SUITE_P(SValTests, SValTest,
                         testing::ValuesIn(allTestClangConfigs()));

} // namespace
} // namespace ento
} // namespace clang

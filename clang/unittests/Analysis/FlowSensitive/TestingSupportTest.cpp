#include "TestingSupport.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Analysis/FlowSensitive/NoopAnalysis.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Testing/ADT/StringMapEntry.h"
#include "llvm/Testing/Support/Error.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace clang;
using namespace dataflow;

namespace {

using ::clang::ast_matchers::functionDecl;
using ::clang::ast_matchers::hasAnyName;
using ::clang::ast_matchers::hasName;
using ::clang::ast_matchers::isDefinition;
using ::clang::dataflow::test::AnalysisInputs;
using ::clang::dataflow::test::AnalysisOutputs;
using ::clang::dataflow::test::checkDataflow;
using ::llvm::IsStringMapEntry;
using ::testing::_;
using ::testing::IsEmpty;
using ::testing::UnorderedElementsAre;

template <typename T>
const FunctionDecl *findTargetFunc(ASTContext &Context, T FunctionMatcher) {
  auto TargetMatcher =
      functionDecl(FunctionMatcher, isDefinition()).bind("target");
  for (const auto &Node : ast_matchers::match(TargetMatcher, Context)) {
    const auto *Func = Node.template getNodeAs<FunctionDecl>("target");
    if (Func == nullptr)
      continue;
    if (Func->isTemplated())
      continue;
    return Func;
  }
  return nullptr;
}

void runTest(
    llvm::StringRef Code, llvm::StringRef TargetName,
    std::function<void(const llvm::DenseMap<const Stmt *, std::string> &)>
        RunChecks) {
  llvm::Annotations AnnotatedCode(Code);
  auto Unit = tooling::buildASTFromCodeWithArgs(
      AnnotatedCode.code(), {"-fsyntax-only", "-std=c++17"});
  auto &Context = Unit->getASTContext();
  const FunctionDecl *Func = findTargetFunc(Context, hasName(TargetName));
  ASSERT_NE(Func, nullptr);

  llvm::Expected<llvm::DenseMap<const Stmt *, std::string>> Mapping =
      test::buildStatementToAnnotationMapping(Func, AnnotatedCode);
  ASSERT_TRUE(static_cast<bool>(Mapping));

  RunChecks(Mapping.get());
}

TEST(BuildStatementToAnnotationMappingTest, ReturnStmt) {
  runTest(R"(
    int target() {
      return 42;
      /*[[ok]]*/
    }
  )",
          "target",
          [](const llvm::DenseMap<const Stmt *, std::string> &Annotations) {
            ASSERT_EQ(Annotations.size(), static_cast<unsigned int>(1));
            EXPECT_TRUE(isa<ReturnStmt>(Annotations.begin()->first));
            EXPECT_EQ(Annotations.begin()->second, "ok");
          });
}

void checkDataflow(
    llvm::StringRef Code,
    ast_matchers::internal::Matcher<FunctionDecl> TargetFuncMatcher,
    std::function<
        void(const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &,
             const AnalysisOutputs &)>
        Expectations) {
  ASSERT_THAT_ERROR(checkDataflow<NoopAnalysis>(
                        AnalysisInputs<NoopAnalysis>(
                            Code, std::move(TargetFuncMatcher),
                            [](ASTContext &Context, Environment &) {
                              return NoopAnalysis(
                                  Context, /*ApplyBuiltinTransfer=*/false);
                            })
                            .withASTBuildArgs({"-fsyntax-only", "-std=c++17"}),
                        /*VerifyResults=*/std::move(Expectations)),
                    llvm::Succeeded());
}

TEST(ProgramPointAnnotations, NoAnnotations) {
  ::testing::MockFunction<void(
      const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &,
      const AnalysisOutputs &)>
      Expectations;

  EXPECT_CALL(Expectations, Call(IsEmpty(), _)).Times(1);

  checkDataflow("void target() {}", hasName("target"),
                Expectations.AsStdFunction());
}

TEST(ProgramPointAnnotations, NoAnnotationsDifferentTarget) {
  ::testing::MockFunction<void(
      const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &,
      const AnalysisOutputs &)>
      Expectations;

  EXPECT_CALL(Expectations, Call(IsEmpty(), _)).Times(1);

  checkDataflow("void target() {}", hasName("target"),
                Expectations.AsStdFunction());
}

TEST(ProgramPointAnnotations, WithProgramPoint) {
  ::testing::MockFunction<void(
      const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &,
      const AnalysisOutputs &)>
      Expectations;

  EXPECT_CALL(
      Expectations,
      Call(UnorderedElementsAre(IsStringMapEntry("program-point", _)), _))
      .Times(1);

  checkDataflow(R"cc(void target() {
                       int n;
                       // [[program-point]]
                     })cc",
                hasName("target"), Expectations.AsStdFunction());
}

TEST(ProgramPointAnnotations, MultipleProgramPoints) {
  ::testing::MockFunction<void(
      const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &,
      const AnalysisOutputs &)>
      Expectations;

  EXPECT_CALL(Expectations,
              Call(UnorderedElementsAre(IsStringMapEntry("program-point-1", _),
                                        IsStringMapEntry("program-point-2", _)),
                   _))
      .Times(1);

  checkDataflow(R"cc(void target(bool b) {
                       if (b) {
                         int n;
                         // [[program-point-1]]
                       } else {
                         int m;
                         // [[program-point-2]]
                       }
                     })cc",
                hasName("target"), Expectations.AsStdFunction());
}

TEST(ProgramPointAnnotations, MultipleFunctionsMultipleProgramPoints) {
  ::testing::MockFunction<void(
      const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &,
      const AnalysisOutputs &)>
      Expectations;

  EXPECT_CALL(Expectations, Call(UnorderedElementsAre(
                                     IsStringMapEntry("program-point-1a", _),
                                     IsStringMapEntry("program-point-1b", _)),
                                 _))
      .Times(1);

  EXPECT_CALL(Expectations, Call(UnorderedElementsAre(
                                     IsStringMapEntry("program-point-2a", _),
                                     IsStringMapEntry("program-point-2b", _)),
                                 _))
      .Times(1);

  checkDataflow(
      R"cc(
        void target1(bool b) {
          if (b) {
            int n;
            // [[program-point-1a]]
          } else {
            int m;
            // [[program-point-1b]]
          }
        }

        void target2(bool b) {
          if (b) {
            int n;
            // [[program-point-2a]]
          } else {
            int m;
            // [[program-point-2b]]
          }
        }
      )cc",
      functionDecl(hasAnyName("target1", "target2")),
      Expectations.AsStdFunction());
}

} // namespace

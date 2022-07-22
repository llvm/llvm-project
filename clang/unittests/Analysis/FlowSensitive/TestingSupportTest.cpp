#include "TestingSupport.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Analysis/FlowSensitive/NoopAnalysis.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Testing/Support/Error.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace clang;
using namespace dataflow;

namespace {

using ::clang::ast_matchers::functionDecl;
using ::clang::ast_matchers::hasName;
using ::clang::ast_matchers::isDefinition;
using ::testing::_;
using ::testing::IsEmpty;
using ::testing::Pair;
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
    llvm::StringRef Code, llvm::StringRef Target,
    std::function<void(llvm::ArrayRef<std::pair<
                           std::string, DataflowAnalysisState<NoopLattice>>>,
                       ASTContext &)>
        Expectations) {
  ASSERT_THAT_ERROR(
      test::checkDataflow<NoopAnalysis>(
          Code, Target,
          [](ASTContext &Context, Environment &) {
            return NoopAnalysis(Context, /*ApplyBuiltinTransfer=*/false);
          },
          std::move(Expectations), {"-fsyntax-only", "-std=c++17"}),
      llvm::Succeeded());
}

TEST(ProgramPointAnnotations, NoAnnotations) {
  ::testing::MockFunction<void(
      llvm::ArrayRef<
          std::pair<std::string, DataflowAnalysisState<NoopLattice>>>,
      ASTContext &)>
      Expectations;

  EXPECT_CALL(Expectations, Call(IsEmpty(), _)).Times(1);

  checkDataflow("void target() {}", "target", Expectations.AsStdFunction());
}

TEST(ProgramPointAnnotations, NoAnnotationsDifferentTarget) {
  ::testing::MockFunction<void(
      llvm::ArrayRef<
          std::pair<std::string, DataflowAnalysisState<NoopLattice>>>,
      ASTContext &)>
      Expectations;

  EXPECT_CALL(Expectations, Call(IsEmpty(), _)).Times(1);

  checkDataflow("void fun() {}", "fun", Expectations.AsStdFunction());
}

TEST(ProgramPointAnnotations, WithCodepoint) {
  ::testing::MockFunction<void(
      llvm::ArrayRef<
          std::pair<std::string, DataflowAnalysisState<NoopLattice>>>,
      ASTContext &)>
      Expectations;

  EXPECT_CALL(Expectations,
              Call(UnorderedElementsAre(Pair("program-point", _)), _))
      .Times(1);

  checkDataflow(R"cc(void target() {
                     int n;
                     // [[program-point]]
                   })cc",
                "target", Expectations.AsStdFunction());
}

TEST(ProgramPointAnnotations, MultipleCodepoints) {
  ::testing::MockFunction<void(
      llvm::ArrayRef<
          std::pair<std::string, DataflowAnalysisState<NoopLattice>>>,
      ASTContext &)>
      Expectations;

  EXPECT_CALL(Expectations,
              Call(UnorderedElementsAre(Pair("program-point-1", _),
                                        Pair("program-point-2", _)),
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
                "target", Expectations.AsStdFunction());
}

} // namespace

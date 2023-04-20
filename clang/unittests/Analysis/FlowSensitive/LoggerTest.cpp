#include "TestingSupport.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Analysis/FlowSensitive/DataflowAnalysis.h"
#include "clang/Analysis/FlowSensitive/DataflowEnvironment.h"
#include "clang/Analysis/FlowSensitive/DataflowLattice.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include <optional>

namespace clang::dataflow::test {
namespace {
using testing::HasSubstr;

struct TestLattice {
  int Elements = 0;
  int Branches = 0;
  int Joins = 0;

  LatticeJoinEffect join(const TestLattice &Other) {
    if (Joins < 3) {
      ++Joins;
      Elements += Other.Elements;
      Branches += Other.Branches;
      return LatticeJoinEffect::Changed;
    }
    return LatticeJoinEffect::Unchanged;
  }
  friend bool operator==(const TestLattice &LHS, const TestLattice &RHS) {
    return std::tie(LHS.Elements, LHS.Branches, LHS.Joins) ==
           std::tie(RHS.Elements, RHS.Branches, RHS.Joins);
  }
};

class TestAnalysis : public DataflowAnalysis<TestAnalysis, TestLattice> {
public:
  using DataflowAnalysis::DataflowAnalysis;

  static TestLattice initialElement() { return TestLattice{}; }
  void transfer(const CFGElement &, TestLattice &L, Environment &E) {
    E.logger().log([](llvm::raw_ostream &OS) { OS << "transfer()"; });
    ++L.Elements;
  }
  void transferBranch(bool Branch, const Stmt *S, TestLattice &L,
                      Environment &E) {
    E.logger().log([&](llvm::raw_ostream &OS) {
      OS << "transferBranch(" << Branch << ")";
    });
    ++L.Branches;
  }
};

class TestLogger : public Logger {
public:
  TestLogger(std::string &S) : OS(S) {}

private:
  llvm::raw_string_ostream OS;

  void beginAnalysis(const ControlFlowContext &,
                     TypeErasedDataflowAnalysis &) override {
    logText("beginAnalysis()");
  }
  void endAnalysis() override { logText("\nendAnalysis()"); }

  void enterBlock(const CFGBlock &B) override {
    OS << "\nenterBlock(" << B.BlockID << ")\n";
  }
  void enterElement(const CFGElement &E) override {
    // we don't want the trailing \n
    std::string S;
    llvm::raw_string_ostream SS(S);
    E.dumpToStream(SS);

    OS << "enterElement(" << llvm::StringRef(S).trim() << ")\n";
  }
  void recordState(TypeErasedDataflowAnalysisState &S) override {
    const TestLattice &L = llvm::any_cast<TestLattice>(S.Lattice.Value);
    OS << "recordState(Elements=" << L.Elements << ", Branches=" << L.Branches
       << ", Joins=" << L.Joins << ")\n";
  }
  /// Records that the analysis state for the current block is now final.
  void blockConverged() override { logText("blockConverged()"); }

  void logText(llvm::StringRef Text) override { OS << Text << "\n"; }
};

AnalysisInputs<TestAnalysis> makeInputs() {
  const char *Code = R"cpp(
int target(bool b, int p, int q) {
  return b ? p : q;    
}
)cpp";
  static const std::vector<std::string> Args = {
      "-fsyntax-only", "-fno-delayed-template-parsing", "-std=c++17"};

  auto Inputs = AnalysisInputs<TestAnalysis>(
      Code, ast_matchers::hasName("target"),
      [](ASTContext &C, Environment &) { return TestAnalysis(C); });
  Inputs.ASTBuildArgs = Args;
  return Inputs;
}

TEST(LoggerTest, Sequence) {
  auto Inputs = makeInputs();
  std::string Log;
  TestLogger Logger(Log);
  Inputs.BuiltinOptions.Log = &Logger;

  ASSERT_THAT_ERROR(checkDataflow<TestAnalysis>(std::move(Inputs),
                                                [](const AnalysisOutputs &) {}),
                    llvm::Succeeded());

  EXPECT_EQ(Log, R"(beginAnalysis()

enterBlock(4)
recordState(Elements=0, Branches=0, Joins=0)
enterElement(b)
transfer()
recordState(Elements=1, Branches=0, Joins=0)
enterElement(b (ImplicitCastExpr, LValueToRValue, _Bool))
transfer()
recordState(Elements=2, Branches=0, Joins=0)

enterBlock(3)
transferBranch(0)
recordState(Elements=2, Branches=1, Joins=0)
enterElement(q)
transfer()
recordState(Elements=3, Branches=1, Joins=0)

enterBlock(2)
transferBranch(1)
recordState(Elements=2, Branches=1, Joins=0)
enterElement(p)
transfer()
recordState(Elements=3, Branches=1, Joins=0)

enterBlock(1)
recordState(Elements=6, Branches=2, Joins=1)
enterElement(b ? p : q)
transfer()
recordState(Elements=7, Branches=2, Joins=1)
enterElement(b ? p : q (ImplicitCastExpr, LValueToRValue, int))
transfer()
recordState(Elements=8, Branches=2, Joins=1)
enterElement(return b ? p : q;)
transfer()
recordState(Elements=9, Branches=2, Joins=1)

enterBlock(0)
recordState(Elements=9, Branches=2, Joins=1)

endAnalysis()
)");
}

TEST(LoggerTest, HTML) {
  auto Inputs = makeInputs();
  std::vector<std::string> Logs;
  auto Logger = Logger::html([&]() {
    Logs.emplace_back();
    return std::make_unique<llvm::raw_string_ostream>(Logs.back());
  });
  Inputs.BuiltinOptions.Log = Logger.get();

  ASSERT_THAT_ERROR(checkDataflow<TestAnalysis>(std::move(Inputs),
                                                [](const AnalysisOutputs &) {}),
                    llvm::Succeeded());

  // Simple smoke tests: we can't meaningfully test the behavior.
  ASSERT_THAT(Logs, testing::SizeIs(1));
  EXPECT_THAT(Logs[0], HasSubstr("function updateSelection")) << "embeds JS";
  EXPECT_THAT(Logs[0], HasSubstr("html {")) << "embeds CSS";
  EXPECT_THAT(Logs[0], HasSubstr("b (ImplicitCastExpr")) << "has CFG elements";
  EXPECT_THAT(Logs[0], HasSubstr("\"B3:1_B3.1\":"))
      << "has analysis point state";
  EXPECT_THAT(Logs[0], HasSubstr("transferBranch(0)")) << "has analysis logs";
  EXPECT_THAT(Logs[0], HasSubstr("LocToVal")) << "has built-in lattice dump";
}

} // namespace
} // namespace clang::dataflow::test

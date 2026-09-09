#include "clang/Analysis/FlowSensitive/TieredSolver.h"

#include <memory>

#include "clang/Analysis/FlowSensitive/Formula.h"
#include "clang/Analysis/FlowSensitive/Solver.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "gtest/gtest.h"

namespace clang::dataflow::test {

namespace {

using Result = Solver::Result;

class MockSolver : public Solver {
public:
  explicit MockSolver(Result Res) : Res(Res) {}

  Result solve(llvm::ArrayRef<const dataflow::Formula *>) override {
    return Res;
  }

  bool reachedLimit() const override {
    return Res.getStatus() == Result::Status::TimedOut;
  }

private:
  Result Res;
};

static llvm::SmallVector<std::unique_ptr<Solver>>
makeMockSolvers(llvm::ArrayRef<Result> Results) {
  llvm::SmallVector<std::unique_ptr<dataflow::Solver>> Solvers;

  for (const auto &Res : Results)
    Solvers.push_back(std::make_unique<MockSolver>(Res));

  return Solvers;
}

TEST(TieredSolverTest, SingleSatisfiableStage) {
  TieredSolver Solver(makeMockSolvers({Result::Satisfiable({{}})}));
  EXPECT_EQ(Solver.solve({}).getStatus(), Result::Status::Satisfiable);
  EXPECT_FALSE(Solver.reachedLimit());
}

TEST(TieredSolverTest, SingleUnsatisfiableStage) {
  TieredSolver Solver(makeMockSolvers({Result::Unsatisfiable()}));
  EXPECT_EQ(Solver.solve({}).getStatus(), Result::Status::Unsatisfiable);
  EXPECT_FALSE(Solver.reachedLimit());
}

TEST(TieredSolverTest, SingleTimedOutStage) {
  TieredSolver Solver(makeMockSolvers({Result::TimedOut()}));
  EXPECT_EQ(Solver.solve({}).getStatus(), Result::Status::TimedOut);
  EXPECT_TRUE(Solver.reachedLimit());
}

TEST(TieredSolverTest, UnsatisfiableThenTimedOut) {
  TieredSolver Solver(
      makeMockSolvers({Result::Unsatisfiable(), Result::TimedOut()}));
  EXPECT_EQ(Solver.solve({}).getStatus(), Result::Status::Unsatisfiable);
  EXPECT_FALSE(Solver.reachedLimit());
}

TEST(TieredSolverTest, TimedOutThenUnsatisfiable) {
  TieredSolver Solver(
      makeMockSolvers({Result::TimedOut(), Result::Unsatisfiable()}));
  EXPECT_EQ(Solver.solve({}).getStatus(), Result::Status::Unsatisfiable);
  EXPECT_FALSE(Solver.reachedLimit());
}

TEST(TieredSolverTest, TwoTimedOutStages) {
  TieredSolver Solver(
      makeMockSolvers({Result::TimedOut(), Result::TimedOut()}));
  EXPECT_EQ(Solver.solve({}).getStatus(), Result::Status::TimedOut);
  EXPECT_TRUE(Solver.reachedLimit());
}

} // namespace

} // namespace clang::dataflow::test

#include "Cancellation.h"
#include "Context.h"
#include "Threading.h"
#include "llvm/Support/Error.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <atomic>
#include <memory>
#include <thread>

namespace clang {
namespace clangd {
namespace {

TEST(CancellationTest, CancellationTest) {
  TaskHandle TH = Task::createHandle();
  WithContext ContextWithCancellation(setCurrentTask(TH));
  EXPECT_FALSE(isCancelled());
  TH->cancel();
  EXPECT_TRUE(isCancelled());
}

TEST(CancellationTest, TaskTestHandleDiesContextLives) {
  llvm::Optional<WithContext> ContextWithCancellation;
  {
    TaskHandle TH = Task::createHandle();
    ContextWithCancellation.emplace(setCurrentTask(TH));
    EXPECT_FALSE(isCancelled());
    TH->cancel();
    EXPECT_TRUE(isCancelled());
  }
  EXPECT_TRUE(isCancelled());
}

TEST(CancellationTest, TaskContextDiesHandleLives) {
  TaskHandle TH = Task::createHandle();
  {
    WithContext ContextWithCancellation(setCurrentTask(TH));
    EXPECT_FALSE(isCancelled());
    TH->cancel();
    EXPECT_TRUE(isCancelled());
  }
  // Still should be able to cancel without any problems.
  TH->cancel();
}

TEST(CancellationTest, CancellationToken) {
  TaskHandle TH = Task::createHandle();
  WithContext ContextWithCancellation(setCurrentTask(TH));
  const auto &CT = getCurrentTask();
  EXPECT_FALSE(CT.isCancelled());
  TH->cancel();
  EXPECT_TRUE(CT.isCancelled());
}

TEST(CancellationTest, AsynCancellationTest) {
  std::atomic<bool> HasCancelled(false);
  Notification Cancelled;
  auto TaskToBeCancelled = [&](ConstTaskHandle CT) {
    WithContext ContextGuard(setCurrentTask(std::move(CT)));
    Cancelled.wait();
    HasCancelled = isCancelled();
  };
  TaskHandle TH = Task::createHandle();
  std::thread AsyncTask(TaskToBeCancelled, TH);
  TH->cancel();
  Cancelled.notify();
  AsyncTask.join();

  EXPECT_TRUE(HasCancelled);
}
} // namespace
} // namespace clangd
} // namespace clang

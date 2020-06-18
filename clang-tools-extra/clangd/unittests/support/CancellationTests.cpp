#include "support/Cancellation.h"
#include "support/Context.h"
#include "support/Threading.h"
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
  auto Task = cancelableTask();
  WithContext ContextWithCancellation(std::move(Task.first));
  EXPECT_FALSE(isCancelled());
  Task.second();
  EXPECT_TRUE(isCancelled());
}

TEST(CancellationTest, CancelerDiesContextLives) {
  llvm::Optional<WithContext> ContextWithCancellation;
  {
    auto Task = cancelableTask();
    ContextWithCancellation.emplace(std::move(Task.first));
    EXPECT_FALSE(isCancelled());
    Task.second();
    EXPECT_TRUE(isCancelled());
  }
  EXPECT_TRUE(isCancelled());
}

TEST(CancellationTest, TaskContextDiesHandleLives) {
  auto Task = cancelableTask();
  {
    WithContext ContextWithCancellation(std::move(Task.first));
    EXPECT_FALSE(isCancelled());
    Task.second();
    EXPECT_TRUE(isCancelled());
  }
  // Still should be able to cancel without any problems.
  Task.second();
}

struct NestedTasks {
  enum { OuterReason = 1, InnerReason = 2 };
  std::pair<Context, Canceler> Outer, Inner;
  NestedTasks() {
    Outer = cancelableTask(OuterReason);
    {
      WithContext WithOuter(Outer.first.clone());
      Inner = cancelableTask(InnerReason);
    }
  }
};

TEST(CancellationTest, Nested) {
  // Cancelling inner task works but leaves outer task unaffected.
  NestedTasks CancelInner;
  CancelInner.Inner.second();
  EXPECT_EQ(NestedTasks::InnerReason, isCancelled(CancelInner.Inner.first));
  EXPECT_FALSE(isCancelled(CancelInner.Outer.first));
  // Cancellation of outer task is inherited by inner task.
  NestedTasks CancelOuter;
  CancelOuter.Outer.second();
  EXPECT_EQ(NestedTasks::OuterReason, isCancelled(CancelOuter.Inner.first));
  EXPECT_EQ(NestedTasks::OuterReason, isCancelled(CancelOuter.Outer.first));
}

TEST(CancellationTest, AsynCancellationTest) {
  std::atomic<bool> HasCancelled(false);
  Notification Cancelled;
  auto TaskToBeCancelled = [&](Context Ctx) {
    WithContext ContextGuard(std::move(Ctx));
    Cancelled.wait();
    HasCancelled = isCancelled();
  };
  auto Task = cancelableTask();
  std::thread AsyncTask(TaskToBeCancelled, std::move(Task.first));
  Task.second();
  Cancelled.notify();
  AsyncTask.join();

  EXPECT_TRUE(HasCancelled);
}
} // namespace
} // namespace clangd
} // namespace clang

//===-- BackgroundQueue.cpp - Task queue for background index -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "index/Background.h"

namespace clang {
namespace clangd {

static std::atomic<bool> PreventStarvation = {false};

void BackgroundQueue::preventThreadStarvationInTests() {
  PreventStarvation.store(true);
}

void BackgroundQueue::work(std::function<void()> OnIdle) {
  while (true) {
    llvm::Optional<Task> Task;
    {
      std::unique_lock<std::mutex> Lock(Mu);
      CV.wait(Lock, [&] { return ShouldStop || !Queue.empty(); });
      if (ShouldStop) {
        Queue.clear();
        CV.notify_all();
        return;
      }
      ++NumActiveTasks;
      std::pop_heap(Queue.begin(), Queue.end());
      Task = std::move(Queue.back());
      Queue.pop_back();
    }

    if (Task->ThreadPri != llvm::ThreadPriority::Default &&
        !PreventStarvation.load())
      llvm::set_thread_priority(Task->ThreadPri);
    Task->Run();
    if (Task->ThreadPri != llvm::ThreadPriority::Default)
      llvm::set_thread_priority(llvm::ThreadPriority::Default);

    {
      std::unique_lock<std::mutex> Lock(Mu);
      if (NumActiveTasks == 1 && Queue.empty() && OnIdle) {
        // We just finished the last item, the queue is going idle.
        Lock.unlock();
        OnIdle();
        Lock.lock();
      }
      assert(NumActiveTasks > 0 && "before decrementing");
      --NumActiveTasks;
    }
    CV.notify_all();
  }
}

void BackgroundQueue::stop() {
  {
    std::lock_guard<std::mutex> QueueLock(Mu);
    ShouldStop = true;
  }
  CV.notify_all();
}

void BackgroundQueue::push(Task T) {
  {
    std::lock_guard<std::mutex> Lock(Mu);
    T.QueuePri = std::max(T.QueuePri, Boosts.lookup(T.Tag));
    Queue.push_back(std::move(T));
    std::push_heap(Queue.begin(), Queue.end());
  }
  CV.notify_all();
}

void BackgroundQueue::append(std::vector<Task> Tasks) {
  {
    std::lock_guard<std::mutex> Lock(Mu);
    for (Task &T : Tasks)
      T.QueuePri = std::max(T.QueuePri, Boosts.lookup(T.Tag));
    std::move(Tasks.begin(), Tasks.end(), std::back_inserter(Queue));
    std::make_heap(Queue.begin(), Queue.end());
  }
  CV.notify_all();
}

void BackgroundQueue::boost(llvm::StringRef Tag, unsigned NewPriority) {
  std::lock_guard<std::mutex> Lock(Mu);
  unsigned &Boost = Boosts[Tag];
  bool Increase = NewPriority > Boost;
  Boost = NewPriority;
  if (!Increase)
    return; // existing tasks unaffected

  unsigned Changes = 0;
  for (Task &T : Queue)
    if (Tag == T.Tag && NewPriority > T.QueuePri) {
      T.QueuePri = NewPriority;
      ++Changes;
    }
  if (Changes)
    std::make_heap(Queue.begin(), Queue.end());
  // No need to signal, only rearranged items in the queue.
}

bool BackgroundQueue::blockUntilIdleForTest(
    llvm::Optional<double> TimeoutSeconds) {
  std::unique_lock<std::mutex> Lock(Mu);
  return wait(Lock, CV, timeoutSeconds(TimeoutSeconds),
              [&] { return Queue.empty() && NumActiveTasks == 0; });
}

} // namespace clangd
} // namespace clang

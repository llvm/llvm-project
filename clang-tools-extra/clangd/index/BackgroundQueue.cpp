//===-- BackgroundQueue.cpp - Task queue for background index -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "index/Background.h"
#include "support/Logger.h"

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
      ++Stat.Active;
      std::pop_heap(Queue.begin(), Queue.end());
      Task = std::move(Queue.back());
      Queue.pop_back();
      notifyProgress();
    }

    if (Task->ThreadPri != llvm::ThreadPriority::Default &&
        !PreventStarvation.load())
      llvm::set_thread_priority(Task->ThreadPri);
    Task->Run();
    if (Task->ThreadPri != llvm::ThreadPriority::Default)
      llvm::set_thread_priority(llvm::ThreadPriority::Default);

    {
      std::unique_lock<std::mutex> Lock(Mu);
      ++Stat.Completed;
      if (Stat.Active == 1 && Queue.empty()) {
        // We just finished the last item, the queue is going idle.
        assert(ShouldStop || Stat.Completed == Stat.Enqueued);
        Stat.LastIdle = Stat.Completed;
        if (OnIdle) {
          Lock.unlock();
          OnIdle();
          Lock.lock();
        }
      }
      assert(Stat.Active > 0 && "before decrementing");
      --Stat.Active;
      notifyProgress();
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
    ++Stat.Enqueued;
    notifyProgress();
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
    Stat.Enqueued += Tasks.size();
    notifyProgress();
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
              [&] { return Queue.empty() && Stat.Active == 0; });
}

void BackgroundQueue::notifyProgress() const {
  dlog("Queue: {0}/{1} ({2} active). Last idle at {3}", Stat.Completed,
       Stat.Enqueued, Stat.Active, Stat.LastIdle);
  if (OnProgress)
    OnProgress(Stat);
}

} // namespace clangd
} // namespace clang

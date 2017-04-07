//===- lld/Core/Parallel.h - Parallel utilities ---------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_CORE_PARALLEL_H
#define LLD_CORE_PARALLEL_H

#include "lld/Core/Instrumentation.h"
#include "lld/Core/LLVM.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/thread.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <stack>

#if defined(_MSC_VER) && LLVM_ENABLE_THREADS
#include <concrt.h>
#include <ppl.h>
#endif

namespace lld {
/// \brief Allows one or more threads to wait on a potentially unknown number of
///   events.
///
/// A latch starts at \p count. inc() increments this, and dec() decrements it.
/// All calls to sync() will block while the count is not 0.
///
/// Calling dec() on a Latch with a count of 0 has undefined behaivor.
class Latch {
  uint32_t _count;
  mutable std::mutex _condMut;
  mutable std::condition_variable _cond;

public:
  explicit Latch(uint32_t count = 0) : _count(count) {}
  ~Latch() { sync(); }

  void inc() {
    std::unique_lock<std::mutex> lock(_condMut);
    ++_count;
  }

  void dec() {
    std::unique_lock<std::mutex> lock(_condMut);
    if (--_count == 0)
      _cond.notify_all();
  }

  void sync() const {
    std::unique_lock<std::mutex> lock(_condMut);
    _cond.wait(lock, [&] {
      return _count == 0;
    });
  }
};

// Classes in this namespace are implementation details of this header.
namespace internal {

/// \brief An abstract class that takes closures and runs them asynchronously.
class Executor {
public:
  virtual ~Executor() = default;
  virtual void add(std::function<void()> func) = 0;
};

#if !defined(LLVM_ENABLE_THREADS) || LLVM_ENABLE_THREADS == 0
class SyncExecutor : public Executor {
public:
  virtual void add(std::function<void()> func) {
    func();
  }
};

inline Executor *getDefaultExecutor() {
  static SyncExecutor exec;
  return &exec;
}
#elif defined(_MSC_VER)
/// \brief An Executor that runs tasks via ConcRT.
class ConcRTExecutor : public Executor {
  struct Taskish {
    Taskish(std::function<void()> task) : _task(task) {}

    std::function<void()> _task;

    static void run(void *p) {
      Taskish *self = static_cast<Taskish *>(p);
      self->_task();
      concurrency::Free(self);
    }
  };

public:
  virtual void add(std::function<void()> func) {
    Concurrency::CurrentScheduler::ScheduleTask(Taskish::run,
        new (concurrency::Alloc(sizeof(Taskish))) Taskish(func));
  }
};

inline Executor *getDefaultExecutor() {
  static ConcRTExecutor exec;
  return &exec;
}
#else
/// \brief An implementation of an Executor that runs closures on a thread pool
///   in filo order.
class ThreadPoolExecutor : public Executor {
public:
  explicit ThreadPoolExecutor(unsigned threadCount =
                                  std::thread::hardware_concurrency())
      : _stop(false), _done(threadCount) {
    // Spawn all but one of the threads in another thread as spawning threads
    // can take a while.
    std::thread([&, threadCount] {
      for (size_t i = 1; i < threadCount; ++i) {
        std::thread([=] {
          work();
        }).detach();
      }
      work();
    }).detach();
  }

  ~ThreadPoolExecutor() override {
    std::unique_lock<std::mutex> lock(_mutex);
    _stop = true;
    lock.unlock();
    _cond.notify_all();
    // Wait for ~Latch.
  }

  void add(std::function<void()> f) override {
    std::unique_lock<std::mutex> lock(_mutex);
    _workStack.push(f);
    lock.unlock();
    _cond.notify_one();
  }

private:
  void work() {
    while (true) {
      std::unique_lock<std::mutex> lock(_mutex);
      _cond.wait(lock, [&] {
        return _stop || !_workStack.empty();
      });
      if (_stop)
        break;
      auto task = _workStack.top();
      _workStack.pop();
      lock.unlock();
      task();
    }
    _done.dec();
  }

  std::atomic<bool> _stop;
  std::stack<std::function<void()>> _workStack;
  std::mutex _mutex;
  std::condition_variable _cond;
  Latch _done;
};

inline Executor *getDefaultExecutor() {
  static ThreadPoolExecutor exec;
  return &exec;
}
#endif

}  // namespace internal

/// \brief Allows launching a number of tasks and waiting for them to finish
///   either explicitly via sync() or implicitly on destruction.
class TaskGroup {
  Latch _latch;

public:
  void spawn(std::function<void()> f) {
    _latch.inc();
    internal::getDefaultExecutor()->add([&, f] {
      f();
      _latch.dec();
    });
  }

  void sync() const { _latch.sync(); }
};

#if !defined(LLVM_ENABLE_THREADS) || LLVM_ENABLE_THREADS == 0
template <class RandomAccessIterator, class Comp>
void parallel_sort(
    RandomAccessIterator start, RandomAccessIterator end,
    const Comp &comp = std::less<
        typename std::iterator_traits<RandomAccessIterator>::value_type>()) {
  std::sort(start, end, comp);
}
#elif defined(_MSC_VER)
// Use ppl parallel_sort on Windows.
template <class RandomAccessIterator, class Comp>
void parallel_sort(
    RandomAccessIterator start, RandomAccessIterator end,
    const Comp &comp = std::less<
        typename std::iterator_traits<RandomAccessIterator>::value_type>()) {
  concurrency::parallel_sort(start, end, comp);
}
#else
namespace detail {
const ptrdiff_t minParallelSize = 1024;

/// \brief Inclusive median.
template <class RandomAccessIterator, class Comp>
RandomAccessIterator medianOf3(RandomAccessIterator start,
                               RandomAccessIterator end, const Comp &comp) {
  RandomAccessIterator mid = start + (std::distance(start, end) / 2);
  return comp(*start, *(end - 1))
         ? (comp(*mid, *(end - 1)) ? (comp(*start, *mid) ? mid : start)
                                   : end - 1)
         : (comp(*mid, *start) ? (comp(*(end - 1), *mid) ? mid : end - 1)
                               : start);
}

template <class RandomAccessIterator, class Comp>
void parallel_quick_sort(RandomAccessIterator start, RandomAccessIterator end,
                         const Comp &comp, TaskGroup &tg, size_t depth) {
  // Do a sequential sort for small inputs.
  if (std::distance(start, end) < detail::minParallelSize || depth == 0) {
    std::sort(start, end, comp);
    return;
  }

  // Partition.
  auto pivot = medianOf3(start, end, comp);
  // Move pivot to end.
  std::swap(*(end - 1), *pivot);
  pivot = std::partition(start, end - 1, [&comp, end](decltype(*start) v) {
    return comp(v, *(end - 1));
  });
  // Move pivot to middle of partition.
  std::swap(*pivot, *(end - 1));

  // Recurse.
  tg.spawn([=, &comp, &tg] {
    parallel_quick_sort(start, pivot, comp, tg, depth - 1);
  });
  parallel_quick_sort(pivot + 1, end, comp, tg, depth - 1);
}
}

template <class RandomAccessIterator, class Comp>
void parallel_sort(
    RandomAccessIterator start, RandomAccessIterator end,
    const Comp &comp = std::less<
        typename std::iterator_traits<RandomAccessIterator>::value_type>()) {
  TaskGroup tg;
  detail::parallel_quick_sort(start, end, comp, tg,
                              llvm::Log2_64(std::distance(start, end)) + 1);
}
#endif

template <class T> void parallel_sort(T *start, T *end) {
  parallel_sort(start, end, std::less<T>());
}

#if !defined(LLVM_ENABLE_THREADS) || LLVM_ENABLE_THREADS == 0
template <class IterTy, class FuncTy>
void parallel_for_each(IterTy Begin, IterTy End, FuncTy Fn) {
  std::for_each(Begin, End, Fn);
}

template <class IndexTy, class FuncTy>
void parallel_for(IndexTy Begin, IndexTy End, FuncTy Fn) {
  for (IndexTy I = Begin; I != End; ++I)
    Fn(I);
}
#elif defined(_MSC_VER)
// Use ppl parallel_for_each on Windows.
template <class IterTy, class FuncTy>
void parallel_for_each(IterTy Begin, IterTy End, FuncTy Fn) {
  concurrency::parallel_for_each(Begin, End, Fn);
}

template <class IndexTy, class FuncTy>
void parallel_for(IndexTy Begin, IndexTy End, FuncTy Fn) {
  concurrency::parallel_for(Begin, End, Fn);
}
#else
template <class IterTy, class FuncTy>
void parallel_for_each(IterTy Begin, IterTy End, FuncTy Fn) {
  // TaskGroup has a relatively high overhead, so we want to reduce
  // the number of spawn() calls. We'll create up to 1024 tasks here.
  // (Note that 1024 is an arbitrary number. This code probably needs
  // improving to take the number of available cores into account.)
  ptrdiff_t TaskSize = std::distance(Begin, End) / 1024;
  if (TaskSize == 0)
    TaskSize = 1;

  TaskGroup Tg;
  while (TaskSize <= std::distance(Begin, End)) {
    Tg.spawn([=, &Fn] { std::for_each(Begin, Begin + TaskSize, Fn); });
    Begin += TaskSize;
  }
  Tg.spawn([=, &Fn] { std::for_each(Begin, End, Fn); });
}

template <class IndexTy, class FuncTy>
void parallel_for(IndexTy Begin, IndexTy End, FuncTy Fn) {
  ptrdiff_t TaskSize = (End - Begin) / 1024;
  if (TaskSize == 0)
    TaskSize = 1;

  TaskGroup Tg;
  IndexTy I = Begin;
  for (; I + TaskSize < End; I += TaskSize) {
    Tg.spawn([=, &Fn] {
      for (IndexTy J = I, E = I + TaskSize; J != E; ++J)
        Fn(J);
    });
  }
  Tg.spawn([=, &Fn] {
    for (IndexTy J = I; J < End; ++J)
      Fn(J);
  });
}
#endif
} // end namespace lld

#endif // LLD_CORE_PARALLEL_H

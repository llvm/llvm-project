//===- lld/Core/TaskGroup.h - Task Group ----------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_CORE_TASKGROUP_H
#define LLD_CORE_TASKGROUP_H

#include "lld/Core/LLVM.h"

#include <condition_variable>
#include <functional>
#include <mutex>

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
    _cond.wait(lock, [&] { return _count == 0; });
  }
};

/// \brief Allows launching a number of tasks and waiting for them to finish
///   either explicitly via sync() or implicitly on destruction.
class TaskGroup {
  Latch _latch;

public:
  void spawn(std::function<void()> f);

  void sync() const { _latch.sync(); }
};
}

#endif

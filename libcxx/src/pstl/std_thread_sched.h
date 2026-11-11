//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef _LIBCPP_SRC_PSTL_STD_THREAD_SCHED_H
#define _LIBCPP_SRC_PSTL_STD_THREAD_SCHED_H

// A PSTL parallelism backend for platforms that only provide the standard std::thread facilities.
// Used by "std::__pstl::__statpart::__apply()" via "Scheduler::apply()".
//
// Design overview:
// - A single Scheduler instance is created on first use and never destroyed.
//   It starts `hardware_concurrency()-1` worker threads that live forever and sleep on an atomic epoch when idle.
// - The worker threads are deliberately never shut down, which keeps the internal state simple.
//   In practice this means that once a parallel algorithm has been called, the process keeps a set of
//   (idle, blocked) threads around until it exits.
// - Scheduler contains a shared board of tasks with a fixed number of slots.
// - apply() acquires a Task, initializes it and opens it for participation by the worker threads.
//   Workers join the Task and run the user function on indices taken from the Task's shared atomic counter.
//   The caller also takes part and then waits for all participants to finish.
//   If the caller couldn't acquire a task slot, it processes the request serially.
// - No dynamic memory is allocated or freed after initialization.
//   If initialization fails or the machine has a single CPU core, everything runs serially.
// - None of this code runs unless PSTL algorithms are used.

#include <__config>

#if _LIBCPP_HAS_THREADS

#  include <__assert>
#  include <array>
#  include <atomic>
#  include <cstddef>
#  include <new>
#  include <thread>

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __pstl::__statpart {

// Assumptions that need to hold for this implementation.
static_assert(std::atomic_size_t::is_always_lock_free);

// Typedefs for apply()'s parameters.
using ctx_t  = void*;
using func_t = void (*)(void*, std::size_t);

// Task represents the user request to apply a function to a range of iterations.
// Tasks are stored inside the Scheduler.
// Multiple threads can participate in working on the same task by picking up the next iteration index and applying the
// function to it.
struct Task {
  using state_t                          = std::atomic_unsigned_lock_free::value_type;
  static constexpr state_t state_unowned = 0;
  static constexpr state_t state_owned   = 1 << 0;
  static constexpr state_t state_open    = 1 << 1;
  static constexpr state_t state_joined  = 1 << 2;

  // State of the task slot:
  // Bit 0: is the slot owned
  // Bit 1: is the slot open for participation
  // Bits 2+: the number of joined workers
  alignas(std::hardware_destructive_interference_size) std::atomic_unsigned_lock_free state{state_unowned};

  // The context pointer to be passed to the function.
  ctx_t ctx = nullptr;

  // The function to be applied to each iteration.
  func_t func = nullptr;

  // The total number of iterations to be processed.
  // This number is atomic to allow workers speculatively skip joining a task.
  std::atomic_size_t iterations{0};

  // The next index of the iteration to be processed.
  alignas(std::hardware_destructive_interference_size) std::atomic_size_t index{0};
};

// Scheduler manages the worker threads and the shared task slots.
class Scheduler {
public:
  // Initializes internal data structures and starts the worker threads.
  Scheduler() noexcept;

  // Destruction is not allowed by design.
  ~Scheduler() = delete;

  // No copying or moving is allowed.
  Scheduler(const Scheduler&)            = delete;
  Scheduler(Scheduler&&)                 = delete;
  Scheduler& operator=(const Scheduler&) = delete;
  Scheduler& operator=(Scheduler&&)      = delete;

  // The entry point to parallel execution.
  // When worker threads are available, applies the function to each of the iteration indices in parallel.
  // When hardware parallelism is not available or in case of initialization failures, applies the function serially.
  void apply(std::size_t iterations, void* ctx, void (*func)(void*, std::size_t)) noexcept;

private:
  // Tries to find and claim ownership of a task slot.
  Task* acquire_task() noexcept;

  // Releases ownership of a task slot.
  void release_task(Task& task) noexcept;

  // Tries to join the task to help with it.
  bool join_task(Task& task) noexcept;

  // Signals that the calling thread no longer participates in the task.
  void leave_task(Task& task) noexcept;

  // Entry point of the worker threads.
  void worker_thread() noexcept;

  // Returns true if there are no worker threads, i.e. all work runs serially.
  bool is_parallelism_disabled() const noexcept;

  // Designate a reasonable number of task slots that can be processed in parallel.
  // There's nothing special about the number, but it should be enough to cover any realistic usage of PSTL.
  // Likely in the overwhelming majority of cases no more than 1 slot will be in use.
  static constexpr std::size_t task_slots_ = 64;

  // The actual number of running worker threads.
  // This number may be lower than the desired thread pool size, in case of a thread startup failure.
  // When this variable is zero, the scheduler doesn't enable its machinery and processes the incoming requests
  // serially.
  std::size_t workers_count_{0};

  // The shared storage of the tasks that can be processed in parallel.
  std::array<Task, task_slots_> tasks_;

  // Wait-wake mechanism.
  // The epoch is incremented each time a new task is opened for participation.
  // Epoch wraparound would not be a problem since only the change itself matters, not the stored value.
  alignas(std::hardware_destructive_interference_size) std::atomic_unsigned_lock_free epoch_{0};
};

// When there are fewer than 2 iterations or no worker threads available, fall back to serial execution.
inline void apply_serial(std::size_t iterations, void* ctx, void (*func)(void*, std::size_t)) noexcept {
  for (std::size_t i = 0; i < iterations; ++i) {
    func(ctx, i);
  }
}

// Participate in working on the task by picking up the next iteration index and applying the function to it.
// Once no iterations are left, close the task slot for participation.
inline void process_task(Task& task) noexcept {
  const std::size_t iterations = task.iterations.load(std::memory_order_relaxed);
  const ctx_t ctx              = task.ctx;
  const func_t func            = task.func;

  std::size_t index;
  while ((index = task.index.fetch_add(1, std::memory_order_relaxed)) < iterations) {
    func(ctx, index);
  }

  // Close the task if it's still open.
  if (task.state.load(std::memory_order_relaxed) & Task::state_open) {
    task.state.fetch_and(~Task::state_open, std::memory_order_relaxed);
  }
}

inline Scheduler::Scheduler() noexcept {
  const std::size_t cpu_threads = std::thread::hardware_concurrency();
  if (cpu_threads < 2) {
    return; // Nothing to do, don't enable parallelism.
  }

  // Start `cpu_threads - 1` worker threads since the calling thread also participates in the work,
  // thus all cpu threads will be occupied.
  const std::size_t max_workers_count = cpu_threads - 1;
  for (; workers_count_ < max_workers_count; ++workers_count_) {
#  if _LIBCPP_HAS_EXCEPTIONS
    try {
#  endif // _LIBCPP_HAS_EXCEPTIONS
      std::thread(&Scheduler::worker_thread, this).detach();
#  if _LIBCPP_HAS_EXCEPTIONS
    } catch (...) {
      break; // Failed to start a worker thread, don't attempt to start any more.
    }
#  endif // _LIBCPP_HAS_EXCEPTIONS
  }
}

inline void Scheduler::worker_thread() noexcept {
  // Worker threads live forever and never return.
  while (true) {
    // Load the epoch before looking for work.
    std::atomic_unsigned_lock_free::value_type epoch = epoch_.load(std::memory_order_acquire);

    // Look at every task slot.
    for (Task& task : tasks_) {
      // Try joining to help with the task.
      if (join_task(task)) {
        process_task(task); // Help with the task.
        leave_task(task);   // Signal that this worker no longer works on it.
      }
    }

    // No more work to do, thus wait until the epoch changes.
    // If the epoch changed while the worker was participating in some tasks, this call will return immediately.
    epoch_.wait(epoch, std::memory_order_relaxed);
  }
}

inline Task* Scheduler::acquire_task() noexcept {
  // Look at every shared task slot and try to claim the first unowned one.
  for (Task& task : tasks_) {
    // Check if this task is currently unowned and try to atomically claim it via CAS.
    Task::state_t expected = Task::state_unowned;
    if (task.state.load(std::memory_order_relaxed) == Task::state_unowned &&
        task.state.compare_exchange_strong(
            expected, Task::state_owned, std::memory_order_acquire, std::memory_order_relaxed)) {
      return &task; // Successfully claimed the task, return it.
    }
  }
  // Failed to acquire any task slot.
  return nullptr;
}

inline void Scheduler::release_task(Task& task) noexcept {
  // To release the task, simply store Task::state_unowned in the task state.
  task.state.store(Task::state_unowned, std::memory_order_release);
}

inline bool Scheduler::join_task(Task& task) noexcept {
  // First look if the task slot is open for participation.
  Task::state_t state = task.state.load(std::memory_order_relaxed);
  while (state & Task::state_open) {
    // If it's open, speculatively look at the number of iterations left to work on and the number of workers already
    // participating. There's no need to join if there's not enough work to do.
    std::size_t iterations = task.iterations.load(std::memory_order_relaxed);
    std::size_t index      = task.index.load(std::memory_order_relaxed);
    // The bits above state_open hold the number of joined workers. Dividing by state_joined extracts it.
    std::size_t joined = state / Task::state_joined;
    if (index >= iterations || joined + 1 >= iterations) {
      // Either the task iterations are already exhausted or the owner plus the joined workers already cover every
      // iteration. This task doesn't need any help.
      return false;
    }
    // Try to atomically join by storing an updated state with an incremented number of joined workers.
    if (task.state.compare_exchange_strong(
            state, state + Task::state_joined, std::memory_order_acquire, std::memory_order_relaxed)) {
      return true; // Successfully joined.
    }
    // The state has changed, try again.
  }
  // The task is not open for joining, bail out.
  return false;
}

inline void Scheduler::leave_task(Task& task) noexcept {
  _LIBCPP_ASSERT_INTERNAL(!(task.state & Task::state_open), "The task must be already closed");

  // Atomically decrement the number of joined threads.
  Task::state_t previous_state = task.state.fetch_sub(Task::state_joined, std::memory_order_release);

  // If we're the last joined thread to leave, notify the current owner of the task about this.
  if (previous_state == (Task::state_owned | Task::state_joined)) {
    task.state.notify_one();
  }
}

inline bool Scheduler::is_parallelism_disabled() const noexcept { return workers_count_ == 0; }

inline void Scheduler::apply(std::size_t iterations, void* ctx, void (*func)(void*, std::size_t)) noexcept {
  if (iterations < 2 || is_parallelism_disabled()) {
    // No worker threads available or not enough iterations requested, fall back to serial execution.
    apply_serial(iterations, ctx, func);
    return;
  }

  // Acquire a slot to publish the new task in.
  Task* task = acquire_task();
  if (!task) {
    // If we cannot acquire a task slot, fall back to serial execution.
    apply_serial(iterations, ctx, func);
    return;
  }

  // Initialize the task that will then be opened for the workers to join in.
  task->ctx  = ctx;
  task->func = func;
  task->iterations.store(iterations, std::memory_order_relaxed);
  task->index.store(0, std::memory_order_relaxed);

  // Open the task slot for others to join.
  task->state.store(Task::state_owned | Task::state_open, std::memory_order_release);

  // Potentially wake up the sleeping workers to participate in the task.
  epoch_.fetch_add(1, std::memory_order_release);
  epoch_.notify_all();

  // Participate in working on the task directly in this thread.
  process_task(*task);

  // Wait until all workers that joined have finished and left.
  Task::state_t state;
  while ((state = task->state.load(std::memory_order_acquire)) != Task::state_owned) {
    task->state.wait(state, std::memory_order_relaxed);
  }

  // Release the task slot so it can be reused.
  release_task(*task);
}

} // namespace __pstl::__statpart
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_HAS_THREADS

#endif // _LIBCPP_SRC_PSTL_STD_THREAD_SCHED_H

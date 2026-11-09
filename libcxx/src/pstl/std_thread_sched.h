//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef _LIBCPP_SRC_PSTL_STD_THREAD_SCHED_H
#define _LIBCPP_SRC_PSTL_STD_THREAD_SCHED_H

#include <__config>

#if _LIBCPP_HAS_THREADS

#  include <__assert>
#  include <__verbose_abort>
#  include <algorithm>
#  include <atomic>
#  include <bit>
#  include <memory>
#  include <new>
#  include <thread>
#  include <type_traits>

_LIBCPP_BEGIN_NAMESPACE_STD
_LIBCPP_BEGIN_EXPLICIT_ABI_ANNOTATIONS
namespace __pstl::__statpart {

// Assumptions that need to hold for this implementation.
static_assert(std::atomic_size_t::is_always_lock_free);
static_assert(std::atomic<std::thread::id>::is_always_lock_free);

// Typedefs for apply()'s parameters.
using ctx_t  = void*;
using func_t = void (*)(void*, std::size_t);

// Task represents the user request to apply a function to a range of iterations.
// It is stored in the stack memory of the caller thread.
// Multiple threads can participate in working on the same task by picking up the next iteration index and applying the
// function to it.
struct Task {
  const ctx_t ctx;              // The context pointer to be passed to the function.
  const func_t func;            // The function to be applied to each iteration.
  const std::size_t iterations; // The total number of iterations to be processed.
  std::atomic_size_t index;     // The next index of the iteration to be processed.
  std::atomic_size_t requests;  // The remaining number of requests to participate in working on this task.
};

// Implementation of "Dynamic Circular Work-Stealing Deque" by David Chase and Yossi Lev, but with a fixed-size buffer.
// The element type is a pointer to Task, nullptr is not a valid value.
struct alignas(std::hardware_destructive_interference_size) TaskQueue {
  constexpr TaskQueue() noexcept         = default;
  TaskQueue(const TaskQueue&)            = delete;
  TaskQueue& operator=(const TaskQueue&) = delete;

  // Allocates the buffer, must be called before the queue is used by any thread.
  // The capacity is rounded up to the next power of two.
  bool init(std::size_t capacity) noexcept {
    capacity = std::bit_ceil(std::max<std::size_t>(capacity, 1));
    elements.reset(new (std::nothrow) Task*[capacity]());
    if (!elements)
      return false;
    mask = capacity - 1;
    return true;
  }

  // Pushes the new task at the bottom of the queue, only called by the owner.
  // Returns false if the queue is full.
  bool push(Task* task) noexcept {
    std::size_t b = bottom.load();
    std::size_t t = top.load();
    if (b - t > mask) // top never passes bottom while the owner is pushing, so b - t is the current size
      return false;
    put(b, task);
    bottom.store(b + 1);
    return true;
  }

  // Pops a task from the bottom of the queue, only called by the owner.
  // Returns nullptr if the queue is empty.
  Task* pop() noexcept {
    std::size_t b = bottom.load();
    --b; // May wrap around, which is fine
    bottom.store(b);
    std::size_t t = top.load();
    if (distance(b, t) >= 0) {
      Task* task = get(b);
      if (distance(b, t) > 0)
        return task;
      if (top.compare_exchange_strong(t, t + 1)) {
        bottom.store(t + 1);
        return task;
      } else {
        bottom.store(t);
        return nullptr;
      }
    } else {
      bottom.store(t);
      return nullptr;
    }
  }

  // Pops a task from the bottom of the queue only if it is the expected one, only called by the owner.
  // Returns nullptr if the queue is empty or if the bottom task is a different one.
  Task* pop_if(Task* expected) noexcept {
    // Only the owner writes the elements and moves the bottom index, so peeking there is safe.
    // If that element has already been stolen, the slot is stale, but then the queue is empty and pop() detects it.
    if (get(bottom.load() - 1) != expected)
      return nullptr;
    return pop();
  }

  // Steals a task from the top of the queue, can be called by any thread.
  // Returns nullptr if the queue is empty. Retries when losing the race for the top element to another thread, which
  // can only happen if that thread made progress, so the loop is lock-free.
  Task* steal() noexcept {
    std::size_t t = top.load();
    while (true) {
      std::size_t b = bottom.load();
      if (distance(b, t) <= 0)
        return nullptr;

      Task* task = get(t);
      if (top.compare_exchange_strong(t, t + 1))
        return task;
      // The CAS failed and loaded the current top into t, retry with it.
    }
  }

private:
  // The indices are allowed to wrap around, so they must be compared via their difference.
  // (a - b) mod 2^N interpreted as signed, correct as long as the real distance is below 2^(N-1).
  static std::ptrdiff_t distance(std::size_t a, std::size_t b) noexcept { return static_cast<std::ptrdiff_t>(a - b); }

  Task* get(std::size_t index) const noexcept {
    // Chase-Lev is inherently racy regarding the elements in the array, wrap access with atomic_ref to remove the UB.
    return std::atomic_ref<Task*>(elements[index & mask]).load(std::memory_order_relaxed);
  }

  void put(std::size_t index, Task* task) noexcept {
    // Chase-Lev is inherently racy regarding the elements in the array, wrap access with atomic_ref to remove the UB.
    std::atomic_ref<Task*>(elements[index & mask]).store(task, std::memory_order_relaxed);
  }

  std::atomic_size_t bottom{0};
  std::atomic_size_t top{0};
  std::unique_ptr<Task*[]> elements;
  std::size_t mask{0};
};

// QueueOwnership represents the ownership of a queue by a thread and the number of nested acquisitions by this thread.
struct QueueOwnership {
  // Identification of a thread that owns a queue.
  // Attempts to acquire ownership must be synchronized via this variable.
  std::atomic<std::thread::id> id = {};

  // Supports nested parallelism: this variable is incremented with each acquisition and decremented with each release.
  // Must only be touched by the thread that owns the queue.
  int acquire_count = 0;
};

// Scheduler manages the worker threads and the queues.
class Scheduler {
public:
  // Allocates internal data structures and starts the worker threads.
  Scheduler() noexcept;

  // Destructor must never be called; doing so causes a verbose abort.
  ~Scheduler();

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
  TaskQueue* acquire_queue(std::thread::id thread_id);
  void release_queue(TaskQueue* queue);
  void worker_thread(size_t worker_index) noexcept;
  bool is_parallelism_disabled() const noexcept;

  // The actual number of running worker threads.
  // This number may be lower than the desired thread pool size, in case of a thread startup failure.
  // When this variable is zero, the scheduler doesn't enable its machinery and processes the incoming requests
  // serially.
  size_t m_workers_count{0};

  // Task queues.
  // [0, m_workers_count) are owned by the worker threads,
  // [m_workers_count, m_queues_count) can be used by the incoming guest threads.
  std::unique_ptr<TaskQueue[]> m_queues;

  // Ownership of the queues in m_queues, this array has the same size.
  std::unique_ptr<QueueOwnership[]> m_queue_owners;

  // The number of task queues, i.e. the sizes of m_queues and m_queue_owners.
  size_t m_queues_count{0};

  // Wait-wake mechanism.
  // The epoch is incremented each time task requests are pushed into a queue.
  std::atomic_unsigned_lock_free m_epoch{0};
};

// When there are fewer than 2 iterations or no worker threads available, fall back to serial execution.
inline void apply_serial(std::size_t iterations, void* ctx, void (*func)(void*, std::size_t)) noexcept {
  for (std::size_t i = 0; i < iterations; ++i) {
    func(ctx, i);
  }
}

// Participate in working on the task by picking up the next iteration index and applying the function to it.
// Once no iterations are left, decrements the task's requests counter to signal that this participant is done.
inline void process_request(Task* task) noexcept {
  const size_t iterations = task->iterations;
  const ctx_t ctx         = task->ctx;
  const func_t func       = task->func;

  size_t index;
  while ((index = task->index.fetch_add(1, std::memory_order_relaxed)) < iterations) {
    func(ctx, index);
  }

  // Signal that this request to participate in the task is finished.
  task->requests.fetch_sub(1, std::memory_order_release);
}

inline Scheduler::Scheduler() noexcept {
  const size_t cpu_threads = std::thread::hardware_concurrency();
  if (cpu_threads < 2) {
    return; // Nothing to do, don't enable parallelism.
  }

  // Start `cpu_threads - 1` worker threads since the calling thread also participates in the work,
  // thus all cpu threads will be occupied.
  const size_t max_workers_count = cpu_threads - 1;

  // Allocate at least cpu_threads * 2 slots in each TaskQueue's buffer.
  // This factor is chosen to allow some nested parallelism, specifically with depth=2.
  const size_t queue_capacity = cpu_threads * 2;

  // Conservatively allow each cpu thread to come in as a guest and publish tasks.
  const size_t guest_queues = cpu_threads;

  // We need one queue per worker thread and one queue per guest thread.
  m_queues_count = max_workers_count + guest_queues;

  // Allocate the queues.
  m_queues.reset(new (std::nothrow) TaskQueue[m_queues_count]);
  if (!m_queues)
    return; // Failed to allocate the queues array, bail out.

  // Allocate the queues' buffers.
  for (size_t i = 0; i < m_queues_count; ++i) {
    if (!m_queues[i].init(queue_capacity))
      return; // Failed to allocate a queue buffer, bail out.
  }

  // Allocate the queue ownership array.
  m_queue_owners.reset(new (std::nothrow) QueueOwnership[m_queues_count]);
  if (!m_queue_owners)
    return; // Failed to allocate the queue ownership array, bail out.

  size_t worker_idx = 0;
  for (; worker_idx < max_workers_count; ++worker_idx) {
    std::thread worker;
#  if _LIBCPP_HAS_EXCEPTIONS
    try {
#  endif // _LIBCPP_HAS_EXCEPTIONS
      worker = std::thread(&Scheduler::worker_thread, this, worker_idx);
#  if _LIBCPP_HAS_EXCEPTIONS
    } catch (...) {
      break; // Failed to start a worker thread, don't attempt to start any more.
    }
#  endif // _LIBCPP_HAS_EXCEPTIONS
    m_queue_owners[worker_idx].id.store(worker.get_id());
    m_queue_owners[worker_idx].acquire_count = 1;
    worker.detach();
  }

  // Finally, enable parallelism by committing the number of workers that have started up.
  m_workers_count = worker_idx;
}

inline Scheduler::~Scheduler() {
  _LIBCPP_VERBOSE_ABORT("The destructor of __pstl::__statpart::Scheduler was called, but it must never be called");
}

inline void Scheduler::worker_thread(size_t worker_index) noexcept {
  // Worker threads live forever and never return.
  while (true) {
    std::atomic_unsigned_lock_free::value_type epoch = m_epoch.load(); // Load the epoch before looking for work
    Task* task                                       = nullptr;

    // The queue owned by this worker thread must be empty at this time, thus there's no reason to try popping from it.

    // Try to steal from other queues in a staggered manner.
    // Loop over each queue except the one owned by this worker.
    for (size_t i = 1; i != m_queues_count; ++i) {
      if ((task = m_queues[(worker_index + i) % m_queues_count].steal())) {
        break;
      }
    }

    if (task) {
      process_request(task); // Participate in working on the task we've just found
    } else {
      m_epoch.wait(epoch); // No work to do, thus wait until the epoch changes
    }
  }
}

inline TaskQueue* Scheduler::acquire_queue(std::thread::id thread_id) {
  // First check if this thread already owns a queue and is coming in via a nested invocation.
  for (size_t i = 0; i < m_queues_count; ++i) {
    if (m_queue_owners[i].id.load() == thread_id) {
      // Yes, this thread already owns a queue, increment the acquire_count and return it.
      m_queue_owners[i].acquire_count++;
      return &m_queues[i];
    }
  }

  // This thread doesn't own a queue, try to find a free one.
  // This thread must be a guest, hence start looking past the queues owned by the workers.
  for (size_t i = m_workers_count; i < m_queues_count; ++i) {
    if (m_queue_owners[i].id.load() == std::thread::id{}) {
      // Nobody owns this queue as of now, try to atomically acquire ownership.
      std::thread::id expected;
      if (m_queue_owners[i].id.compare_exchange_strong(expected, thread_id)) {
        // Acquired by a new guest thread, set acquire_count to 1 and return this queue.
        m_queue_owners[i].acquire_count = 1;
        return &m_queues[i];
      }
    }
  }

  // Unable to acquire a queue for this thread.
  return nullptr;
}

inline void Scheduler::release_queue(TaskQueue* queue) {
  size_t idx = static_cast<size_t>(queue - m_queues.get());
  m_queue_owners[idx].acquire_count--;
  if (m_queue_owners[idx].acquire_count == 0) {
    m_queue_owners[idx].id.store(std::thread::id{});
  }
}

inline bool Scheduler::is_parallelism_disabled() const noexcept { return m_workers_count == 0; }

inline void Scheduler::apply(std::size_t iterations, void* ctx, void (*func)(void*, std::size_t)) noexcept {
  if (iterations < 2 || is_parallelism_disabled()) {
    // No worker threads available or not enough iterations requested, fall back to serial execution.
    apply_serial(iterations, ctx, func);
    return;
  }

  // Identify ourselves.
  // We can be either a guest thread or a worker thread re-entering via nested parallelism.
  std::thread::id thread_id = std::this_thread::get_id();

  // Acquire a queue for this thread.
  TaskQueue* queue = acquire_queue(thread_id);
  if (queue == nullptr) {
    // If we cannot acquire a queue, fallback to serial execution.
    apply_serial(iterations, ctx, func);
    return;
  }

  // Split the workload for potentially (m_workers_count + 1) participants (workers + caller thread).
  size_t num_requests = std::min(m_workers_count + 1, iterations);

  // Create a task whose requests will then be pushed in the queue and stolen by the workers.
  Task root_task{.ctx = ctx, .func = func, .iterations = iterations, .index = 0, .requests = num_requests};

  // Push all but one request to the queue, and process the last one directly in this thread.
  // If the queue is full (e.g. due to deep nested parallelism), the caller thread will process all remaining requests
  // directly.
  while (num_requests > 1) {
    if (!queue->push(&root_task)) {
      break;
    }
    --num_requests;
  }

  // Potentially wake up the sleeping workers to process the requests in the queue.
  m_epoch.fetch_add(1);
  m_epoch.notify_all();

  // Process the remaining request(s) directly in this thread.
  while (num_requests > 0) {
    process_request(&root_task);
    --num_requests;
  }

  // Wait until all requests to participate in root_task are processed.
  // While waiting, only participate in the requests to root_task that might still be left in this queue.
  // This loop deliberately does not try to steal requests from other queues and help with progress of their tasks,
  // since this might end up processing the requests of root_task in an interleaved manner, which is not allowed by the
  // Standard and can cause a deadlock inside the user code.
  while (root_task.requests.load(std::memory_order_acquire) != 0) {
    if (Task* task = queue->pop_if(&root_task)) {
      // Participate in root_task, i.e. decrement its requests count, as the iterations must already be exhausted.
      _LIBCPP_ASSERT_INTERNAL(task == &root_task, "Expected to pop the root_task from the queue");
      _LIBCPP_ASSERT_INTERNAL(task->index.load() >= task->iterations, "The iterations must already be exhausted");
      process_request(task);
    } else {
      // Wait for other threads to finish participating in root_task.
      std::this_thread::yield();
    }
  }

  // Release the queue for this thread.
  release_queue(queue);
}

} // namespace __pstl::__statpart
_LIBCPP_END_EXPLICIT_ABI_ANNOTATIONS
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_HAS_THREADS

#endif // _LIBCPP_SRC_PSTL_STD_THREAD_SCHED_H

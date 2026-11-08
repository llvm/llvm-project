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

#  include <algorithm>
#  include <atomic>
#  include <bit>
#  include <latch>
#  include <memory>
#  include <new>
#  include <thread>
#  include <type_traits>

_LIBCPP_BEGIN_NAMESPACE_STD
_LIBCPP_BEGIN_EXPLICIT_ABI_ANNOTATIONS
namespace __pstl::__statpart {

// Assumptions that need to hold for this implementation
static_assert(std::atomic_size_t::is_always_lock_free);
static_assert(std::atomic<std::thread::id>::is_always_lock_free);

// Task represent the user request to apply a function to a range of iterations.
// It is stored in the stack memory of the caller thread.
// Multiple threads can participate in working on the same task by picking up the next iteration index and applying the
// function to it.
struct Task {
  void* ctxt;                       // The context pointer to be passed to the function.
  void (*func)(void*, std::size_t); // The function to be applied to each iteration.
  std::size_t iterations;           // The total number of iterations to be processed.
  std::atomic_size_t index;         // The next index of the iteration to be processed.
  std::latch requests;              // Latch used to synchronize the caller with the worker threads.
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
    // Chase-Lev is inherently racy with regards to the elements of the array, wrap access with atomic_ref to quiet TSAN
    return std::atomic_ref<Task*>(elements[index & mask]).load(std::memory_order_relaxed);
  }

  void put(std::size_t index, Task* task) noexcept {
    // Chase-Lev is inherently racy with regards to the elements of the array, wrap access with atomic_ref to quiet TSAN
    std::atomic_ref<Task*>(elements[index & mask]).store(task, std::memory_order_relaxed);
  }

  std::atomic_size_t bottom{0};
  std::atomic_size_t top{0};
  std::unique_ptr<Task*[]> elements;
  std::size_t mask{0};
};

// QueueID represents the ownership of a queue and the number of nested acquisitions.
struct QueueID {
  std::atomic<std::thread::id> id = {};
  std::atomic_size_t aquire_count = 0;
};

// Scheduler manages the worker threads and the queues.
class Scheduler {
public:
  Scheduler() noexcept;

  // No copying or moving is allowed.
  Scheduler(const Scheduler&)            = delete;
  Scheduler(Scheduler&&)                 = delete;
  Scheduler& operator=(const Scheduler&) = delete;
  Scheduler& operator=(Scheduler&&)      = delete;

  void apply(std::size_t iterations, void* ctxt, void (*func)(void*, std::size_t)) noexcept;

private:
  // The actual number of running worker threads.
  // This number may be lower than the desired thread pool sizee, in case of a thread startup failure.
  // When this variable is zero, the scheduler doesn't enable its machinery.
  size_t m_workers_count{0};

  std::unique_ptr<std::thread[]> m_workers;

  std::unique_ptr<TaskQueue[]> m_queues;

  std::unique_ptr<QueueID[]> m_queue_ids;

  size_t m_queues_count{0};

  // Wait-wake mechanism.
  // The epoch is monotonically incremented after each time task invitations are pushed into a task queue.
  // Wrapping is allowed and harmless.
  std::atomic_unsigned_lock_free m_epoch{0};

  TaskQueue* acquire_queue(std::thread::id thread_id);
  void release_queue(TaskQueue* queue);
  void worker_thread(size_t worker_num) noexcept;
  bool is_parallelism_enabled() const noexcept;
};

// For cases when the number of iterations is less than 2 or when there are no worker threads available,
// fallback to serial execution.
inline void apply_serial(std::size_t iterations, void* ctxt, void (*func)(void*, std::size_t)) noexcept {
  for (std::size_t i = 0; i < iterations; ++i) {
    func(ctxt, i);
  }
}

// Participate in working on the task by picking up the next iteration index and applying the function to it.
// When all iterations are processed, the latch is counted down to signal completion.
inline void process_request(Task* task) noexcept {
  size_t const iterations                = task->iterations;
  void* const ctxt                       = task->ctxt;
  void (*const func)(void*, std::size_t) = task->func;

  size_t index;
  while ((index = task->index.fetch_add(1, std::memory_order_relaxed)) < iterations) {
    func(ctxt, index);
  }

  task->requests.count_down();
}

inline Scheduler::Scheduler() noexcept {
  const size_t cpu_threads = std::thread::hardware_concurrency();
  if (cpu_threads < 2) {
    return; // Nothing to do, don't enable parallelism
  }

  // Allocate `cpu_threads - 1` worker threads since the guest thread will also participate in the work, thus all cpu
  // threads will be occupied.
  const size_t max_workers_count = cpu_threads - 1;

  // Allocate at least cpu_threads * 2 slots in the TaskQueue's buffer.
  // This factor is chosen to allow some nested parallelism, specifically with depth=2.
  const size_t queue_capacity = cpu_threads * 2;

  // We need a queue per each worker thread and a queue per cpu thread for the guest threads.
  m_queues_count = max_workers_count + cpu_threads;

  // Allocate the queues
  m_queues.reset(new (std::nothrow) TaskQueue[m_queues_count]);
  if (!m_queues)
    return; // Failed to allocate the queues array, bail out

  // Allocate the queues' buffers
  for (size_t i = 0; i < m_queues_count; ++i) {
    if (!m_queues[i].init(queue_capacity))
      return; // Failed to allocate a queue buffer, bail out.
  }

  // Allocate the queue identifiers array
  m_queue_ids.reset(new (std::nothrow) QueueID[m_queues_count]);
  if (!m_queue_ids)
    return; // Failed to allocate the queue identifiers buffer, bail out.

  // Allocate the array of worker threads
  m_workers.reset(new (std::nothrow) std::thread[max_workers_count]);
  if (!m_workers)
    return; // Failed to allocate the threads array, bail out.

  size_t worker_idx = 0;
  for (; worker_idx < max_workers_count; ++worker_idx) {
    std::thread worker;
#  if _LIBCPP_HAS_EXCEPTIONS
    try {
#  endif // _LIBCPP_HAS_EXCEPTIONS
      worker = std::thread(&Scheduler::worker_thread, this, worker_idx);
#  if _LIBCPP_HAS_EXCEPTIONS
    } catch (...) {
      break; // Failed to start a worker thread, don't attempt to start any more
    }
#  endif // _LIBCPP_HAS_EXCEPTIONS
    m_queue_ids[worker_idx].id.store(worker.get_id());
    m_queue_ids[worker_idx].aquire_count.store(1);
    m_workers[worker_idx] = std::move(worker);
  }

  // Finally, enable parallelism by committing the number of workers that have started up.
  m_workers_count = worker_idx;
}

inline void Scheduler::worker_thread(size_t worker_index) noexcept {
  while (true) {
    std::atomic_unsigned_lock_free::value_type epoch = m_epoch.load(); // Load the epoch before looking for work
    Task* task                                       = nullptr;

    // Try to pop from own queue
    if (!(task = m_queues[worker_index].pop())) {
      // If there's no tasks in our queue - try to steal from other queues in a staggared manner
      for (size_t i = 1; i != m_queues_count; ++i) // Loop over each queue but the one owned by the worker
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
  for (size_t i = 0; i < m_queues_count; ++i) {
    std::thread::id id = m_queue_ids[i].id.load();
    if (id == thread_id) {
      bool is_guest = i >= m_workers_count;
      if (is_guest) {
        m_queue_ids[i].aquire_count.fetch_add(1);
      }
      return &m_queues[i];
    }
  }
  for (size_t i = 0; i < m_queues_count; ++i) {
    std::thread::id id = m_queue_ids[i].id.load();
    if (id == std::thread::id()) {
      std::thread::id expected;
      if (m_queue_ids[i].id.compare_exchange_strong(expected, thread_id)) {
        // Acquired by a new guest thread => set aquire_count to 1
        m_queue_ids[i].aquire_count.store(1);
        return &m_queues[i];
      }
    }
  }
  return nullptr;
}

inline void Scheduler::release_queue(TaskQueue* queue) {
  size_t idx    = static_cast<size_t>(queue - m_queues.get());
  bool is_guest = idx >= m_workers_count;
  if (is_guest) {
    if (m_queue_ids[idx].aquire_count.fetch_sub(1) == 1) {
      m_queue_ids[idx].id.store(std::thread::id());
    }
  }
}

inline bool Scheduler::is_parallelism_enabled() const noexcept { return m_workers_count != 0; }

// The entrypoint to parallel execution.
// Apply the function to the range of iterations in parallel using worker threads.
// The caller thread also participates in processing the workload.
inline void Scheduler::apply(std::size_t iterations, void* ctxt, void (*func)(void*, std::size_t)) noexcept {
  if (iterations < 2 || !is_parallelism_enabled()) {
    // No worker threads available or not enough iteration requested, fallback to serial execution
    apply_serial(iterations, ctxt, func);
    return;
  }

  // Identify ourselves.
  // We can be either a guest thread or worker thread via nested parallelism.
  std::thread::id thread_id = std::this_thread::get_id();

  // Acquire a queue for this thread.
  TaskQueue* queue = acquire_queue(thread_id);
  if (queue == nullptr) {
    // If we cannot acquire a queue, fallback to serial execution.
    apply_serial(iterations, ctxt, func);
    return;
  }

  // Split the workload for potentially (m_workers_count + 1) participants (workers + caller thread).
  size_t num_requests = std::min(m_workers_count + 1, iterations);

  // Create a task that will be published in the queue and then stolen by the workers.
  Task root_task{
      .ctxt       = ctxt,                             //
      .func       = func,                             //
      .iterations = iterations,                       //
      .index      = 0,                                //
      .requests{static_cast<ptrdiff_t>(num_requests)} //
  };

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

  // Process the last request directly in this thread.
  while (num_requests > 0) {
    process_request(&root_task);
    --num_requests;
  }

  // Once the original task is completed, wait for all other requests to be completed by the workers.
  // While waiting, the caller thread can also participate in processing other requests from the queue or steal from
  // other queues.
  while (!root_task.requests.try_wait()) {
    Task* task = nullptr;

    // Try to pop from own queue
    if (!(task = queue->pop())) {
      // Try to steal from other queues
      for (size_t i = 0; i != m_queues_count; ++i) {
        if (&m_queues[i] != queue && (task = m_queues[i].steal())) {
          break;
        }
      }
    }

    if (task) {
      // Do the work
      process_request(task);
    } else {
      // Wait a bit and try again
      std::this_thread::yield();
    }
  }

  // Synchronize with the workers to ensure nobody holds a reference to the task.
  root_task.requests.wait();

  // Release the queue for this thread.
  release_queue(queue);
}

} // namespace __pstl::__statpart
_LIBCPP_END_EXPLICIT_ABI_ANNOTATIONS
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_HAS_THREADS

#endif // _LIBCPP_SRC_PSTL_STD_THREAD_SCHED_H

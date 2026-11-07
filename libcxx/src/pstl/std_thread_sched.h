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
#include <algorithm>
#include <atomic>
#include <bit>
#include <latch>
#include <memory>
#include <mutex>
#include <new>
#include <thread>
#include <type_traits>

// #define WITH_LOGGING 1
#ifdef WITH_LOGGING
#  include <stdio.h>
#endif

_LIBCPP_BEGIN_NAMESPACE_STD
_LIBCPP_BEGIN_EXPLICIT_ABI_ANNOTATIONS
namespace __pstl::__statpart {

#ifdef WITH_LOGGING
[[clang::no_destroy]] static std::mutex LOG_MUTEX;
#  define LOG(...)                                                                                                     \
    do {                                                                                                               \
      std::lock_guard lock{LOG_MUTEX};                                                                                 \
      fprintf(stderr, __VA_ARGS__);                                                                                    \
    } while (0)
#else
#  define LOG(...)                                                                                                     \
    do {                                                                                                               \
    } while (0)
#endif

struct Task;

// Implementation of "Dynamic Circular Work-Stealing Deque" by David Chase and Yossi Lev, but with a fixed-size buffer.
// The element type is a pointer to Task, nullptr is not a valid value.
struct alignas(std::hardware_destructive_interference_size) SPMCQueue {
  SPMCQueue() noexcept                   = default;
  SPMCQueue(const SPMCQueue&)            = delete;
  SPMCQueue& operator=(const SPMCQueue&) = delete;

  // Allocates the buffer, must be called before the queue is used by any thread.
  // The capacity is rounded up to the next power of two.
  void init(std::size_t capacity) {
    // TODO: make this function nothrow and return a boolean to signal malloc failures
    capacity = std::bit_ceil(std::max<std::size_t>(capacity, 1));
    elements = std::make_unique<Task*[]>(capacity);
    mask     = capacity - 1;
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
  // Returns nullptr if the queue is empty or if the steal fails due to contention.
  Task* steal() noexcept {
    std::size_t t = top.load();
    std::size_t b = bottom.load();
    if (distance(b, t) <= 0)
      return nullptr;

    Task* task = get(t);
    if (top.compare_exchange_strong(t, t + 1))
      return task;
    else
      return nullptr;
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

  static_assert(std::atomic_size_t::is_always_lock_free);
  std::atomic_size_t bottom{0};
  std::atomic_size_t top{0};
  std::unique_ptr<Task*[]> elements;
  std::size_t mask{0};
};

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

// QueueID represents the ownership of a queue and the number of nested acquisitions.
struct QueueID {
  static_assert(std::atomic<std::thread::id>::is_always_lock_free);
  std::atomic<std::thread::id> id = {};
  std::atomic_size_t aquire_count = 0;
};

// Sched is the scheduler that manages the worker threads and the queues.
struct Sched {
private:
  std::unique_ptr<std::thread[]> m_workers;
  std::unique_ptr<SPMCQueue[]> m_queues;
  std::unique_ptr<QueueID[]> m_queue_ids;
  size_t m_workers_count;
  size_t m_queues_count;
  std::atomic_unsigned_lock_free m_epoch{0}; // Wait-wake mechanism

  void worker_thread(size_t worker_num) noexcept;

public:
  Sched() {
    size_t cpu_threads = std::thread::hardware_concurrency();
    if (cpu_threads == 0)
      cpu_threads = 1; // safe fallback

    m_queues_count = cpu_threads * 2;
    m_queues       = std::make_unique<SPMCQueue[]>(m_queues_count);
    for (size_t i = 0; i < m_queues_count; ++i)
      m_queues[i].init(cpu_threads * 2);
    m_queue_ids = std::make_unique<QueueID[]>(m_queues_count);

    m_workers_count = cpu_threads - 1;
    m_workers       = std::make_unique<std::thread[]>(m_workers_count);

    for (size_t i = 0; i < m_workers_count; ++i) {
      std::thread worker(&Sched::worker_thread, this, i);
      m_queue_ids[i].id.store(worker.get_id());
      m_queue_ids[i].aquire_count.store(1);
      m_workers[i] = std::move(worker);
    }
  }

  void apply(std::size_t iterations, void* ctxt, void (*func)(void*, std::size_t)) noexcept;

  SPMCQueue* acquire_queue(std::thread::id thread_id) {
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

  void release_queue(SPMCQueue* queue) {
    size_t idx    = static_cast<size_t>(queue - m_queues.get());
    bool is_guest = idx >= m_workers_count;
    if (is_guest) {
      if (m_queue_ids[idx].aquire_count.fetch_sub(1) == 1) {
        m_queue_ids[idx].id.store(std::thread::id());
      }
    }
  }
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

inline void Sched::worker_thread(size_t worker_num) noexcept {
  const size_t queues_count    = m_queues_count;
  constexpr int max_spin_count = 16;
  int spin_count               = 0;

  while (true) {
    auto epoch = m_epoch.load();
    Task* task = nullptr;

    // 1st - try to pop from own queue
    if ((task = m_queues[worker_num].pop())) {
      LOG("Worker %zu popped task %p from own queue\n", worker_num, task);
    } else {
      // 2nd - try to steal from other queues
      for (size_t i = 0; i != queues_count; ++i) {
        size_t idx = (worker_num + i) % queues_count;
        if ((task = m_queues[idx].steal())) {
          LOG("Worker %zu stole task %p from queue %zu\n", worker_num, task, idx);
          break;
        }
      }
    }

    if (task) {
      // Do the work
      process_request(task);
      spin_count = 0;
    } else {
      // Wait and sleep
      if (spin_count < max_spin_count) {
        ++spin_count;
        std::this_thread::yield();
      } else {
        LOG("Worker %zu is going to sleep\n", worker_num);
        m_epoch.wait(epoch);
        spin_count = 0;
        LOG("Worker %zu woke up\n", worker_num);
      }
    }
  }
}

// The entrypoint to parallel execution.
// Apply the function to the range of iterations in parallel using worker threads.
// The caller thread also participates in processing the workload.
inline void Sched::apply(std::size_t iterations, void* ctxt, void (*func)(void*, std::size_t)) noexcept {
  if (iterations < 2 || m_workers_count == 0) {
    // No worker threads available or not enough iteration requested, fallback to serial execution
    apply_serial(iterations, ctxt, func);
    return;
  }

  // Identify ourselves.
  // We can be either a guest thread or worker thread via nested parallelism.
  std::thread::id thread_id = std::this_thread::get_id();

  // Acquire a queue for this thread.
  SPMCQueue* queue = acquire_queue(thread_id);
  if (queue == nullptr) {
    // If we cannot acquire a queue, fallback to serial execution.
    apply_serial(iterations, ctxt, func);
    return;
  }

  // Split the workload for potentially (workers_count + 1) participants (workers + caller thread).
  size_t num_requests = std::min(m_workers_count + 1, iterations);

  // Create a task that will be published in the queue and then stolen by the workers.
  Task root_task{
      .ctxt       = ctxt,                             //
      .func       = func,                             //
      .iterations = iterations,                       //
      .index      = 0,                                //
      .requests{static_cast<ptrdiff_t>(num_requests)} //
  };

  LOG("New root task %p of %zu iterations with %zu requests\n", &root_task, iterations, num_requests);

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
    LOG("Guest thread is directly processing task %p, num_requests left: %zu\n", &root_task, num_requests);
    process_request(&root_task);
    --num_requests;
  }

  // Once the original task is completed, wait for all other requests to be completed by the workers.
  // While waiting, the caller thread can also participate in processing other requests from the queue or steal from
  // other queues.
  while (!root_task.requests.try_wait()) {
    Task* task = nullptr;

    // 1st - try to pop from own queue
    if ((task = queue->pop())) {
      LOG("Guest thread popped task %p from own queue\n", task);
    } else {
      // 2nd - try to steal from other queues
      for (size_t i = 0; i != m_queues_count; ++i) {
        if (&m_queues[i] != queue && (task = m_queues[i].steal())) {
          LOG("Guest stole task %p from queue %zu\n", task, i);
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

#endif // _LIBCPP_SRC_PSTL_STD_THREAD_SCHED_H

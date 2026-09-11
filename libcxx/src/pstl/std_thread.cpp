//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <__algorithm/max.h>
#include <__atomic/aliases.h>
#include <__atomic/atomic.h>
#include <__atomic/atomic_ref.h>
#include <__config>
#include <__mutex/lock_guard.h>
#include <__mutex/mutex.h>
#include <__mutex/once_flag.h>
#include <__mutex/unique_lock.h>
#include <__new/interference_size.h>
#include <__pstl/backends/std_thread.h>
#include <__thread/this_thread.h>
#include <__thread/thread.h>
#include <__type_traits/is_trivial.h>
#include <latch>

// #define WITH_LOGGING 1
#ifdef WITH_LOGGING
#  include <stdio.h>
#endif

_LIBCPP_BEGIN_NAMESPACE_STD
_LIBCPP_BEGIN_EXPLICIT_ABI_ANNOTATIONS
namespace __pstl::__std_thread {

#if defined(_AIX) && !defined(__64BIT__)
// on AIX (32-bit):
// c++/v1/__atomic/support/c11.h:83:10: error: large atomic operation may incur significant performance penalty;
// the access size (8 bytes) exceeds the max lock-free size (4 bytes) [-Werror,-Watomic-alignment]
// Sequential dummy implementation for now.
void __apply(size_t __iterations, void* __context, void (*__func)(void* __context, size_t __iteration)) noexcept {
  for (std::size_t i = 0; i < __iterations; ++i) {
    __func(__context, i);
  }
}
#else

#  ifdef WITH_LOGGING
[[clang::no_destroy]] static std::mutex LOG_MUTEX;
#    define LOG(...)                                                                                                   \
      do {                                                                                                             \
        std::lock_guard lock{LOG_MUTEX};                                                                               \
        fprintf(stderr, __VA_ARGS__);                                                                                  \
      } while (0)
#  else
#    define LOG(...)                                                                                                   \
      do {                                                                                                             \
      } while (0)
#  endif

struct Task;

// Implementation of "Dynamic Circular Work-Stealing Deque" by David Chase and Yossi Lev.
// The element type is a pointer to Task, nullptr is not a valid value.
struct alignas(std::hardware_destructive_interference_size) SPMCQueue {
  static constexpr unsigned log_initial_size = 6; // 64 elements by default

  SPMCQueue() noexcept : bottom(0), top(0), buffer(alloc_buffer(log_initial_size)) {}
  SPMCQueue(const SPMCQueue&)            = delete;
  SPMCQueue& operator=(const SPMCQueue&) = delete;
  ~SPMCQueue() { ::operator delete(buffer); }

  // Pushes the new task at the bottom of the queue, only called by the owner.
  // Can return false if the allocation of grown buffer fails.
  bool push(Task* task) noexcept {
    std::uint64_t b    = bottom.load();
    std::uint64_t t    = top.load();
    std::uint64_t size = b - t;
    if (size >= buffer_size(buffer) - 1) {
      Buffer* grown = grow_buffer(buffer, b, t);
      if (grown == nullptr) {
        return false;
      }
      Buffer* current;
      {
        std::lock_guard lock{mut};
        current = std::exchange(buffer, grown);
      }
      if (current->ref_count.fetch_sub(1) == 1) {
        ::operator delete(current);
      }
    }
    put(buffer, b, task);
    bottom.store(b + 1);
    return true;
  }

  // Pops a task from the bottom of the queue, only called by the owner.
  // Returns nullptr if the queue is empty.
  Task* pop() noexcept {
    std::uint64_t b = bottom.load();
    if (b == 0)
      return nullptr;
    --b;
    bottom.store(b);
    std::uint64_t t = top.load();
    if (t <= b) {
      Task* task = get(buffer, b);
      if (b > t)
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
    std::uint64_t t = top.load();
    std::uint64_t b = bottom.load();
    if (t >= b)
      return nullptr;

    Buffer* current;
    {
      std::lock_guard lock{mut};
      current = buffer;
      current->ref_count.fetch_add(1);
    }
    Task* task = get(current, t);
    if (current->ref_count.fetch_sub(1) == 1) {
      ::operator delete(current);
    }

    if (top.compare_exchange_strong(t, t + 1))
      return task;
    else
      return nullptr;

    return nullptr;
  }

private:
  struct Buffer {
    std::uint32_t log_size;
    std::atomic_int32_t ref_count;
  };

  static Buffer* alloc_buffer(unsigned log_size) noexcept {
    std::size_t size  = 1Z << log_size;
    std::size_t bytes = sizeof(SPMCQueue) + sizeof(Task*) * size;
    Buffer* buffer    = static_cast<Buffer*>(::operator new(bytes, std::nothrow));
    if (buffer == nullptr)
      return nullptr;
    buffer->log_size  = log_size;
    buffer->ref_count = 1;
    return buffer;
  }

  static Buffer* grow_buffer(Buffer* existing, std::uint64_t bottom, std::uint64_t top) noexcept {
    Buffer* grown = alloc_buffer(existing->log_size + 1);
    if (grown == nullptr)
      return nullptr;
    for (std::uint64_t i = top; i != bottom; ++i) {
      put(grown, i, get(existing, i));
    }
    return grown;
  }

  static std::size_t buffer_size(Buffer* buffer) noexcept { return 1Z << buffer->log_size; }

  static Task* get(Buffer* buffer, std::size_t index) noexcept {
    Task** elements  = reinterpret_cast<Task**>(reinterpret_cast<std::byte*>(buffer) + sizeof(Buffer));
    std::size_t mask = (1Z << buffer->log_size) - 1;
    // Chase-Lev is inherently racy with regards to the elements of the array, wrap access with atomic_ref to quiet TSAN
    return std::atomic_ref<Task*>(elements[index & mask]).load(std::memory_order_relaxed);
  }

  static void put(Buffer* buffer, std::size_t index, Task* task) noexcept {
    Task** elements  = reinterpret_cast<Task**>(reinterpret_cast<std::byte*>(buffer) + sizeof(Buffer));
    std::size_t mask = (1Z << buffer->log_size) - 1;
    // Chase-Lev is inherently racy with regards to the elements of the array, wrap access with atomic_ref to quiet TSAN
    std::atomic_ref<Task*>(elements[index & mask]).store(task, std::memory_order_relaxed);
  }

  std::atomic_uint64_t bottom;
  std::atomic_uint64_t top;
  Buffer* buffer;
  std::mutex mut;
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
  std::mutex m_workers_sleep_mutex;
  std::condition_variable m_workers_sleep_cv;

  void worker_thread(size_t worker_num) noexcept;

public:
  Sched() {
    size_t cpu_threads = std::thread::hardware_concurrency();
    if (cpu_threads == 0)
      cpu_threads = 1; // safe fallback

    m_queues_count = cpu_threads * 2;
    m_queues       = std::make_unique<SPMCQueue[]>(m_queues_count);
    m_queue_ids    = std::make_unique<QueueID[]>(m_queues_count);

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

  size_t num_workers() const noexcept { return m_workers_count; }

  void wake_workers() noexcept { m_workers_sleep_cv.notify_all(); }

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

static Sched* g_sched = nullptr;
static std::once_flag g_sched_once_flag;

static Sched* get_sched() {
  std::call_once(g_sched_once_flag, []() { g_sched = new Sched(); });
  return g_sched;
}

// For cases when the number of iterations is less than 2 or when there are no worker threads available,
// fallback to serial execution.
static void apply_serial(std::size_t iterations, void* ctxt, void (*func)(void*, std::size_t)) noexcept {
  for (std::size_t i = 0; i < iterations; ++i) {
    func(ctxt, i);
  }
}

// Participate in working on the task by picking up the next iteration index and applying the function to it.
// When all iterations are processed, the latch is counted down to signal completion.
static void process_request(Task* task) noexcept {
  size_t const iterations                = task->iterations;
  void* const ctxt                       = task->ctxt;
  void (*const func)(void*, std::size_t) = task->func;

  size_t index;
  while ((index = task->index.fetch_add(1, std::memory_order_relaxed)) < iterations) {
    func(ctxt, index);
  }

  task->requests.count_down();
}

void Sched::worker_thread(size_t worker_num) noexcept {
  const size_t queues_count    = m_queues_count;
  constexpr int max_spin_count = 16;
  int spin_count               = 0;

  while (true) {
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
    } else {
      // Wait and sleep
      if (spin_count < max_spin_count) {
        ++spin_count;
        std::this_thread::yield();
      } else {
        LOG("Worker %zu is going to sleep\n", worker_num);
        std::unique_lock lock{m_workers_sleep_mutex};
        m_workers_sleep_cv.wait(lock);
        spin_count = 0;
        LOG("Worker %zu woke up\n", worker_num);
      }
    }
  }
}

// The entrypoint to parallel execution.
// Apply the function to the range of iterations in parallel using worker threads.
// The caller thread also participates in processing the workload.
void Sched::apply(std::size_t iterations, void* ctxt, void (*func)(void*, std::size_t)) noexcept {
  if (m_workers_count == 0) {
    // No worker threads available, fallback to serial execution
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
  // Also support an edge-case when the queue fails to grow, in this case the caller thread will process all remaining
  // requests directly.
  while (num_requests > 1) {
    if (!queue->push(&root_task)) {
      break;
    }
    --num_requests;
  }

  // Potentially wake up the sleeping workers to process the requests in the queue.
  wake_workers();

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

// The entrypoint to parallel execution.
void __apply(size_t __iterations, void* __context, void (*__func)(void* __context, size_t __iteration)) noexcept {
  if (__iterations < 2) {
    apply_serial(__iterations, __context, __func);
    return;
  }
  Sched* sched = get_sched();
  sched->apply(__iterations, __context, __func);
}

#endif

// This actually doesn't strictly needs to be in the implementation file.
__chunk_partitions __partition_chunks(ptrdiff_t element_count) noexcept {
  __chunk_partitions partitions;
  partitions.__chunk_count_      = std::max<ptrdiff_t>(1, element_count / 256);
  partitions.__chunk_size_       = element_count / partitions.__chunk_count_;
  partitions.__first_chunk_size_ = element_count - (partitions.__chunk_count_ - 1) * partitions.__chunk_size_;
  if (partitions.__chunk_count_ == 0 && element_count > 0)
    partitions.__chunk_count_ = 1;
  return partitions;
}

} // namespace __pstl::__std_thread
_LIBCPP_END_EXPLICIT_ABI_ANNOTATIONS
_LIBCPP_END_NAMESPACE_STD

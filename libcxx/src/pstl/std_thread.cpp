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

struct alignas(std::hardware_destructive_interference_size) SPMCQueue {
  static constexpr unsigned log_initial_size = 6; // 64 elements by default

  SPMCQueue() noexcept : bottom(0), top(0), buffer(alloc_buffer(log_initial_size)) {}
  SPMCQueue(const SPMCQueue&)            = delete;
  SPMCQueue& operator=(const SPMCQueue&) = delete;
  ~SPMCQueue() { ::operator delete(buffer); }

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
    return elements[index & mask];
  }

  static void put(Buffer* buffer, std::size_t index, Task* task) noexcept {
    Task** elements        = reinterpret_cast<Task**>(reinterpret_cast<std::byte*>(buffer) + sizeof(Buffer));
    std::size_t mask       = (1Z << buffer->log_size) - 1;
    elements[index & mask] = task;
  }

  std::atomic_uint64_t bottom;
  std::atomic_uint64_t top;
  Buffer* buffer;
  std::mutex mut;
};

struct Task {
  void* ctxt;
  void (*func)(void*, std::size_t);
  std::size_t iterations;
  std::atomic_size_t index;
  std::latch requests;
};

struct QueueID {
  static_assert(std::atomic<std::thread::id>::is_always_lock_free);
  std::atomic<std::thread::id> id = {};
  std::atomic_size_t aquire_count = 0;
};

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

static void apply_serial(std::size_t iterations, void* ctxt, void (*func)(void*, std::size_t)) noexcept {
  for (std::size_t i = 0; i < iterations; ++i) {
    func(ctxt, i);
  }
}

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

void Sched::apply(std::size_t iterations, void* ctxt, void (*func)(void*, std::size_t)) noexcept {
  if (m_workers_count == 0) {
    // No worker threads available, fallback to serial execution
    apply_serial(iterations, ctxt, func);
    return;
  }

  std::thread::id thread_id = std::this_thread::get_id();
  SPMCQueue* queue          = acquire_queue(thread_id);
  if (queue == nullptr) {
    apply_serial(iterations, ctxt, func);
    return;
  }

  size_t num_requests = std::min(m_workers_count + 1, iterations);
  Task root_task{
      .ctxt       = ctxt,                             //
      .func       = func,                             //
      .iterations = iterations,                       //
      .index      = 0,                                //
      .requests{static_cast<ptrdiff_t>(num_requests)} //
  };

  LOG("New root task %p of %zu iterations with %zu requests\n", &root_task, iterations, num_requests);

  // Push all but one request to the queue, and process the last one directly in this thread.
  while (num_requests > 1) {
    if (!queue->push(&root_task)) {
      break;
    }
    --num_requests;
  }

  wake_workers();

  // Process the last request directly in this thread
  while (num_requests > 0) {
    LOG("Guest thread is directly processing task %p, num_requests left: %zu\n", &root_task, num_requests);
    process_request(&root_task);
    --num_requests;
  }

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

  root_task.requests.wait();

  release_queue(queue);
}

void __apply(size_t __iterations, void* __context, void (*__func)(void* __context, size_t __iteration)) noexcept {
  if (__iterations < 2) {
    apply_serial(__iterations, __context, __func);
    return;
  }
  Sched* sched = get_sched();
  sched->apply(__iterations, __context, __func);
}

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

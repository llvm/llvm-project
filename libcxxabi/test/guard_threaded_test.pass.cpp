//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03
// UNSUPPORTED: libcxxabi-no-threads, libcxxabi-no-exceptions

#define TESTING_CXA_GUARD
#include "../src/cxa_guard_impl.h"
#include <unordered_map>
#include <thread>
#include <atomic>
#include <array>
#include <cassert>
#include <memory>
#include <vector>


using namespace __cxxabiv1;

enum class InitResult {
  COMPLETE,
  PERFORMED,
  WAITED,
  ABORTED
};
constexpr InitResult COMPLETE = InitResult::COMPLETE;
constexpr InitResult PERFORMED = InitResult::PERFORMED;
constexpr InitResult WAITED = InitResult::WAITED;
constexpr InitResult ABORTED = InitResult::ABORTED;


template <class Impl, class GuardType, class Init>
InitResult check_guard(GuardType *g, Init init) {
  uint8_t *first_byte = reinterpret_cast<uint8_t*>(g);
  if (std::__libcpp_atomic_load(first_byte, std::_AO_Acquire) == 0) {
    Impl impl(g);
    if (impl.cxa_guard_acquire() == INIT_IS_PENDING) {
#ifndef LIBCXXABI_HAS_NO_EXCEPTIONS
      try {
#endif
        init();
        impl.cxa_guard_release();
        return PERFORMED;
#ifndef LIBCXXABI_HAS_NO_EXCEPTIONS
      } catch (...) {
        impl.cxa_guard_abort();
        return ABORTED;
      }
#endif
    }
    return WAITED;
  }
  return COMPLETE;
}


template <class GuardType, class Impl>
struct FunctionLocalStatic {
  FunctionLocalStatic() { reset(); }
  FunctionLocalStatic(FunctionLocalStatic const&) = delete;

  template <class InitFunc>
  InitResult access(InitFunc&& init) {
    ++waiting_threads;
    auto res = check_guard<Impl>(&guard_object, init);
    --waiting_threads;
    ++result_counts[static_cast<int>(res)];
    return res;
  }

  struct Accessor {
    explicit Accessor(FunctionLocalStatic& obj) : this_obj(&obj) {}

    template <class InitFn>
    void operator()(InitFn && fn) const {
      this_obj->access(std::forward<InitFn>(fn));
    }
  private:
    FunctionLocalStatic *this_obj;
  };

  Accessor get_access() {
    return Accessor(*this);
  }

  void reset() {
    guard_object = 0;
    waiting_threads.store(0);
    for (auto& counter : result_counts) {
      counter.store(0);
    }
  }

  int get_count(InitResult I) const {
    return result_counts[static_cast<int>(I)].load();
  }
  int num_completed() const {
    return get_count(COMPLETE) + get_count(PERFORMED) + get_count(WAITED);
  }
  int num_waiting() const {
    return waiting_threads.load();
  }

private:
  GuardType guard_object;
  std::atomic<int> waiting_threads;
  std::array<std::atomic<int>, 4> result_counts;
  static_assert(static_cast<int>(ABORTED) == 3, "only 4 result kinds expected");
};

struct ThreadGroup {
  ThreadGroup() = default;
  ThreadGroup(ThreadGroup const&) = delete;

  template <class ...Args>
  void Create(Args&& ...args) {
    threads.emplace_back(std::forward<Args>(args)...);
  }

  void JoinAll() {
    for (auto& t : threads) {
      t.join();
    }
  }

private:
  std::vector<std::thread> threads;
};

struct Barrier {
  explicit Barrier(int n) : m_wait_for(n) { reset(); }
  Barrier(Barrier const&) = delete;

  void wait() {
    ++m_entered;
    while (m_entered.load() < m_wait_for) {
      std::this_thread::yield();
    }
    assert(m_entered.load() == m_wait_for);
    ++m_exited;
  }

  int num_waiting() const {
    return m_entered.load() - m_exited.load();
  }

  void reset() {
    m_entered.store(0);
    m_exited.store(0);
  }
private:
  const int m_wait_for;
  std::atomic<int> m_entered;
  std::atomic<int> m_exited;
};

struct Notification {
  Notification() { reset(); }
  Notification(Notification const&) = delete;

  int num_waiting() const {
    return m_waiting.load();
  }

  void wait() {
    if (m_cond.load())
      return;
    ++m_waiting;
    while (!m_cond.load()) {
      std::this_thread::yield();
    }
    --m_waiting;
  }

  void notify() {
    m_cond.store(true);
  }

  template <class Cond>
  void notify_when(Cond &&c) {
    if (m_cond.load())
      return;
    while (!c()) {
      std::this_thread::yield();
    }
    m_cond.store(true);
  }

  void reset() {
    m_cond.store(0);
    m_waiting.store(0);
  }
private:
  std::atomic<bool> m_cond;
  std::atomic<int> m_waiting;
};


template <class GuardType, class Impl>
void test_free_for_all() {
  const int num_waiting_threads = 10; // one initializing thread, 10 waiters.

  FunctionLocalStatic<GuardType, Impl> test_obj;

  Barrier start_init_barrier(num_waiting_threads);
  bool already_init = false;
  ThreadGroup threads;
  for (int i=0; i < num_waiting_threads; ++i) {
    threads.Create([&]() {
      start_init_barrier.wait();
      test_obj.access([&]() {
        assert(!already_init);
        already_init = true;
      });
    });
  }

  // wait for the other threads to finish initialization.
  threads.JoinAll();

  assert(test_obj.get_count(PERFORMED) == 1);
  assert(test_obj.get_count(COMPLETE) + test_obj.get_count(WAITED) == 9);
}

template <class GuardType, class Impl>
void test_waiting_for_init() {
    const int num_waiting_threads = 10; // one initializing thread, 10 waiters.

    Notification init_pending;
    Notification init_barrier;
    FunctionLocalStatic<GuardType, Impl> test_obj;
    auto access_fn = test_obj.get_access();

    ThreadGroup threads;
    threads.Create(access_fn,
      [&]() {
        init_pending.notify();
        init_barrier.wait();
      }
    );
    init_pending.wait();

    assert(test_obj.num_waiting() == 1);

    for (int i=0; i < num_waiting_threads; ++i) {
      threads.Create(access_fn, []() { assert(false); });
    }
    // unblock the initializing thread
    init_barrier.notify_when([&]() {
      return test_obj.num_waiting() == num_waiting_threads + 1;
    });

    // wait for the other threads to finish initialization.
    threads.JoinAll();

    assert(test_obj.get_count(PERFORMED) == 1);
    assert(test_obj.get_count(WAITED) == 10);
    assert(test_obj.get_count(COMPLETE) == 0);
}


template <class GuardType, class Impl>
void test_aborted_init() {
  const int num_waiting_threads = 10; // one initializing thread, 10 waiters.

  Notification init_pending;
  Notification init_barrier;
  FunctionLocalStatic<GuardType, Impl> test_obj;
  auto access_fn = test_obj.get_access();

  ThreadGroup threads;
  threads.Create(access_fn,
                 [&]() {
                   init_pending.notify();
                   init_barrier.wait();
                   throw 42;
                 }
  );
  init_pending.wait();

  assert(test_obj.num_waiting() == 1);

  bool already_init = false;
  for (int i=0; i < num_waiting_threads; ++i) {
    threads.Create(access_fn, [&]() {
      assert(!already_init);
      already_init = true;
    });
  }
  // unblock the initializing thread
  init_barrier.notify_when([&]() {
    return test_obj.num_waiting() == num_waiting_threads + 1;
  });

  // wait for the other threads to finish initialization.
  threads.JoinAll();

  assert(test_obj.get_count(ABORTED) == 1);
  assert(test_obj.get_count(PERFORMED) == 1);
  assert(test_obj.get_count(WAITED) == 9);
  assert(test_obj.get_count(COMPLETE) == 0);
}


template <class GuardType, class Impl>
void test_completed_init() {
  const int num_waiting_threads = 10; // one initializing thread, 10 waiters.

  Notification init_barrier;
  FunctionLocalStatic<GuardType, Impl> test_obj;

  test_obj.access([]() {});
  assert(test_obj.num_waiting() == 0);
  assert(test_obj.num_completed() == 1);
  assert(test_obj.get_count(PERFORMED) == 1);

  auto access_fn = test_obj.get_access();
  ThreadGroup threads;
  for (int i=0; i < num_waiting_threads; ++i) {
    threads.Create(access_fn, []() {
      assert(false);
    });
  }

  // wait for the other threads to finish initialization.
  threads.JoinAll();

  assert(test_obj.get_count(ABORTED) == 0);
  assert(test_obj.get_count(PERFORMED) == 1);
  assert(test_obj.get_count(WAITED) == 0);
  assert(test_obj.get_count(COMPLETE) == 10);
}

template <class Impl>
void test_impl() {
  {
    test_free_for_all<uint32_t, Impl>();
    test_free_for_all<uint32_t, Impl>();
  }
  {
    test_waiting_for_init<uint32_t, Impl>();
    test_waiting_for_init<uint64_t, Impl>();
  }
  {
    test_aborted_init<uint32_t, Impl>();
    test_aborted_init<uint64_t, Impl>();
  }
  {
    test_completed_init<uint32_t, Impl>();
    test_completed_init<uint64_t, Impl>();
  }
}

void test_all_impls() {
  using MutexImpl = SelectImplementation<Implementation::GlobalLock>::type;

  // Attempt to test the Futex based implementation if it's supported on the
  // target platform.
  using RealFutexImpl = SelectImplementation<Implementation::Futex>::type;
  using FutexImpl = typename std::conditional<
      PlatformSupportsFutex(),
      RealFutexImpl,
      MutexImpl
  >::type;

  // Run each test 5 times to help TSAN catch bugs.
  const int num_runs = 5;
  for (int i=0; i < num_runs; ++i) {
    test_impl<MutexImpl>();
    if (PlatformSupportsFutex())
      test_impl<FutexImpl>();
  }
}

// A dummy
template <bool Dummy = true>
void test_futex_syscall() {
  if (!PlatformSupportsFutex())
    return;
  int lock1 = 0;
  int lock2 = 0;
  int lock3 = 0;
  std::thread waiter1([&]() {
    int expect = 0;
    PlatformFutexWait(&lock1, expect);
    assert(lock1 == 1);
  });
  std::thread waiter2([&]() {
    int expect = 0;
    PlatformFutexWait(&lock2, expect);
    assert(lock2 == 2);
  });
  std::thread waiter3([&]() {
    int expect = 42; // not the value
    PlatformFutexWait(&lock3, expect); // doesn't block
  });
  std::thread waker([&]() {
    lock1 = 1;
    PlatformFutexWake(&lock1);
    lock2 = 2;
    PlatformFutexWake(&lock2);
  });
  waiter1.join();
  waiter2.join();
  waiter3.join();
  waker.join();
}

int main() {
  // Test each multi-threaded implementation with real threads.
  test_all_impls();
  // Test the basic sanity of the futex syscall wrappers.
  test_futex_syscall();
}

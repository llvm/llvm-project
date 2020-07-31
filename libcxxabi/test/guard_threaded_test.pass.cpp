//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03
// UNSUPPORTED: libcxxabi-no-threads
// UNSUPPORTED: no-exceptions

#define TESTING_CXA_GUARD
#include "../src/cxa_guard_impl.h"
#include <unordered_map>
#include <thread>
#include <atomic>
#include <array>
#include <cassert>
#include <memory>
#include <vector>

#include "test_macros.h"


using namespace __cxxabiv1;

// Misc test configuration. It's used to tune the flakyness of the test.
// ThreadsPerTest - The number of threads used
constexpr int ThreadsPerTest = 10;
// The number of instances of a test to run concurrently.
constexpr int ConcurrentRunsPerTest = 10;
// The number of times to rerun each test.
constexpr int TestSamples = 50;



void BusyWait() {
  std::this_thread::yield();
}

void YieldAfterBarrier() {
  std::this_thread::sleep_for(std::chrono::nanoseconds(10));
  std::this_thread::yield();
}

struct Barrier {
  explicit Barrier(int n) : m_threads(n), m_remaining(n) { }
  Barrier(Barrier const&) = delete;
  Barrier& operator=(Barrier const&) = delete;

  void arrive_and_wait() const {
    --m_remaining;
    while (m_remaining.load()) {
      BusyWait();
    }
  }

  void arrive_and_drop()  const {
    --m_remaining;
  }

  void wait_for_threads(int n) const {
    while ((m_threads - m_remaining.load()) < n) {
      std::this_thread::yield();
    }
  }

private:
  const int m_threads;
  mutable std::atomic<int> m_remaining;
};


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
#ifndef TEST_HAS_NO_EXCEPTIONS
      try {
#endif
        init();
        impl.cxa_guard_release();
        return PERFORMED;
#ifndef TEST_HAS_NO_EXCEPTIONS
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
  FunctionLocalStatic() {}
  FunctionLocalStatic(FunctionLocalStatic const&) = delete;

  template <class InitFunc>
  InitResult access(InitFunc&& init) {
    auto res = check_guard<Impl>(&guard_object, init);
    ++result_counts[static_cast<int>(res)];
    return res;
  }

  template <class InitFn>
  struct AccessCallback {
    void operator()() const { this_obj->access(init); }

    FunctionLocalStatic *this_obj;
    InitFn init;
  };

  template <class InitFn, class Callback = AccessCallback< InitFn >  >
  Callback access_callback(InitFn init) {
    return Callback{this, init};
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
  GuardType guard_object = {};
  std::atomic<int> waiting_threads{0};
  std::array<std::atomic<int>, 4> result_counts{};
  static_assert(static_cast<int>(ABORTED) == 3, "only 4 result kinds expected");
};

struct ThreadGroup {
  ThreadGroup() = default;
  ThreadGroup(ThreadGroup const&) = delete;

  template <class ...Args>
  void Create(Args&& ...args) {
    threads.emplace_back(std::forward<Args>(args)...);
  }

  template <class Callback>
  void CreateThreadsWithBarrier(int N, Callback cb) {
    auto start = std::make_shared<Barrier>(N + 1);
    for (int I=0; I < N; ++I) {
      Create([start, cb]() {
        start->arrive_and_wait();
        cb();
      });
    }
    start->arrive_and_wait();
  }

  void JoinAll() {
    for (auto& t : threads) {
      t.join();
    }
  }

private:
  std::vector<std::thread> threads;
};


template <class GuardType, class Impl>
void test_free_for_all(int num_waiters) {
  FunctionLocalStatic<GuardType, Impl> test_obj;

  ThreadGroup threads;

  bool already_init = false;
  threads.CreateThreadsWithBarrier(num_waiters,
    test_obj.access_callback([&]() {
      assert(!already_init);
      already_init = true;
    })
  );

  // wait for the other threads to finish initialization.
  threads.JoinAll();

  assert(test_obj.get_count(PERFORMED) == 1);
  assert(test_obj.get_count(COMPLETE) + test_obj.get_count(WAITED) == num_waiters - 1);
}

template <class GuardType, class Impl>
void test_waiting_for_init(int num_waiters) {
    FunctionLocalStatic<GuardType, Impl> test_obj;

    ThreadGroup threads;

    Barrier start_init(2);
    threads.Create(test_obj.access_callback(
      [&]() {
        start_init.arrive_and_wait();
        // Take our sweet time completing the initialization...
        //
        // There's a race condition between the other threads reaching the
        // start_init barrier, and them actually hitting the cxa guard.
        // But we're trying to test the waiting logic, we want as many
        // threads to enter the waiting loop as possible.
        YieldAfterBarrier();
      }
    ));
    start_init.wait_for_threads(1);

    threads.CreateThreadsWithBarrier(num_waiters,
        test_obj.access_callback([]() { assert(false); })
    );
    // unblock the initializing thread
    start_init.arrive_and_drop();

    // wait for the other threads to finish initialization.
    threads.JoinAll();

    assert(test_obj.get_count(PERFORMED) == 1);
    assert(test_obj.get_count(ABORTED) == 0);
    assert(test_obj.get_count(COMPLETE) + test_obj.get_count(WAITED) == num_waiters);
}


template <class GuardType, class Impl>
void test_aborted_init(int num_waiters) {
  FunctionLocalStatic<GuardType, Impl> test_obj;

  Barrier start_init(2);
  ThreadGroup threads;
  threads.Create(test_obj.access_callback(
    [&]() {
      start_init.arrive_and_wait();
      YieldAfterBarrier();
      throw 42;
    })
  );
  start_init.wait_for_threads(1);

  bool already_init = false;
  threads.CreateThreadsWithBarrier(num_waiters,
      test_obj.access_callback([&]() {
        assert(!already_init);
        already_init = true;
      })
    );
  // unblock the initializing thread
  start_init.arrive_and_drop();

  // wait for the other threads to finish initialization.
  threads.JoinAll();

  assert(test_obj.get_count(ABORTED) == 1);
  assert(test_obj.get_count(PERFORMED) == 1);
  assert(test_obj.get_count(WAITED) + test_obj.get_count(COMPLETE) == num_waiters - 1);
}


template <class GuardType, class Impl>
void test_completed_init(int num_waiters) {

  FunctionLocalStatic<GuardType, Impl> test_obj;

  test_obj.access([]() {}); // initialize the object
  assert(test_obj.num_waiting() == 0);
  assert(test_obj.num_completed() == 1);
  assert(test_obj.get_count(PERFORMED) == 1);

  ThreadGroup threads;
  threads.CreateThreadsWithBarrier(num_waiters,
      test_obj.access_callback([]() { assert(false); })
  );
  // wait for the other threads to finish initialization.
  threads.JoinAll();

  assert(test_obj.get_count(ABORTED) == 0);
  assert(test_obj.get_count(PERFORMED) == 1);
  assert(test_obj.get_count(WAITED) == 0);
  assert(test_obj.get_count(COMPLETE) == num_waiters);
}

template <class Impl>
void test_impl() {
  using TestFn = void(*)(int);
  TestFn TestList[] = {
    test_free_for_all<uint32_t, Impl>,
    test_free_for_all<uint32_t, Impl>,
    test_waiting_for_init<uint32_t, Impl>,
    test_waiting_for_init<uint64_t, Impl>,
    test_aborted_init<uint32_t, Impl>,
    test_aborted_init<uint64_t, Impl>,
    test_completed_init<uint32_t, Impl>,
    test_completed_init<uint64_t, Impl>
  };

  for (auto test_func : TestList) {
      ThreadGroup test_threads;
      test_threads.CreateThreadsWithBarrier(ConcurrentRunsPerTest, [=]() {
        for (int I = 0; I < TestSamples; ++I) {
          test_func(ThreadsPerTest);
        }
      });
      test_threads.JoinAll();
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

  test_impl<MutexImpl>();
  if (PlatformSupportsFutex())
    test_impl<FutexImpl>();
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

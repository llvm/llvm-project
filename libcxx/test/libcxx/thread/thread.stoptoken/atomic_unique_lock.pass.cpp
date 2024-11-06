//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-threads
// UNSUPPORTED: c++03, c++11, c++14, c++17
// XFAIL: availability-synchronization_library-missing
// ADDITIONAL_COMPILE_FLAGS: -Wno-private-header

#include <__stop_token/atomic_unique_lock.h>
#include <atomic>
#include <cassert>
#include <chrono>
#include <thread>

#include "make_test_thread.h"
#include "test_macros.h"

template <uint8_t LockBit>
void test() {
  using Lock = std::__atomic_unique_lock<uint8_t, LockBit>;

  // lock on constructor
  {
    std::atomic<uint8_t> state{0};
    Lock l(state);
    assert(l.__owns_lock());
  }

  // always give up locking
  {
    std::atomic<uint8_t> state{0};
    Lock l(state, [](auto const&) { return true; });
    assert(!l.__owns_lock());
  }

  // test overload that has custom state after lock
  {
    std::atomic<uint8_t> state{0};
    auto neverGiveUpLocking = [](auto const&) { return false; };
    auto stateAfter         = [](auto) { return uint8_t{255}; };
    Lock l(state, neverGiveUpLocking, stateAfter, std::memory_order_acq_rel);
    assert(l.__owns_lock());
    assert(state.load() == 255);
  }

  // lock and unlock
  {
    std::atomic<uint8_t> state{0};
    Lock l(state);
    assert(l.__owns_lock());

    l.__unlock();
    assert(!l.__owns_lock());

    l.__lock();
    assert(l.__owns_lock());
  }

  // lock blocking
  {
    std::atomic<uint8_t> state{0};
    int i = 0;
    Lock l1(state);

    auto thread1 = support::make_test_thread([&] {
      std::this_thread::sleep_for(std::chrono::milliseconds{10});
      i = 5;
      l1.__unlock();
    });

    Lock l2(state);
    // l2's lock has to wait for l1's unlocking
    assert(i == 5);

    thread1.join();
  }
}

int main(int, char**) {
  test<1 << 0>();
  test<1 << 1>();
  test<1 << 2>();
  test<1 << 3>();
  test<1 << 4>();
  test<1 << 5>();
  test<1 << 6>();
  test<1 << 7>();
  return 0;
}

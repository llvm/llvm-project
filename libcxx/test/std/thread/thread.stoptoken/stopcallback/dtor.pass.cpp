//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads
// UNSUPPORTED: libcpp-has-no-experimental-stop_token
// UNSUPPORTED: c++03, c++11, c++14, c++17
// XFAIL: availability-synchronization_library-missing

// ~stop_callback();

#include <atomic>
#include <cassert>
#include <chrono>
#include <functional>
#include <memory>
#include <optional>
#include <stop_token>
#include <type_traits>
#include <utility>
#include <vector>

#include "make_test_thread.h"
#include "test_macros.h"

struct CallbackHolder;

struct DeleteHolder {
  CallbackHolder& holder_;
  void operator()() const;
};

struct CallbackHolder {
  std::unique_ptr<std::stop_callback<DeleteHolder>> callback_;
};

void DeleteHolder::operator()() const { holder_.callback_.reset(); }

int main(int, char**) {
  // Unregisters the callback from the owned stop state, if any
  {
    std::stop_source ss;
    bool called = false;

    {
      std::stop_callback sc(ss.get_token(), [&] { called = true; });
    }
    ss.request_stop();
    assert(!called);
  }

  // The destructor does not block waiting for the execution of another
  // callback registered by an associated stop_callback.
  {
    std::stop_source ss;

    std::atomic<int> startedIndex    = 0;
    std::atomic<bool> callbackFinish = false;

    std::optional<std::stop_callback<std::function<void()>>> sc1(std::in_place, ss.get_token(), [&] {
      startedIndex = 1;
      startedIndex.notify_all();
      callbackFinish.wait(false);
    });

    std::optional<std::stop_callback<std::function<void()>>> sc2(std::in_place, ss.get_token(), [&] {
      startedIndex = 2;
      startedIndex.notify_all();
      callbackFinish.wait(false);
    });

    auto thread = support::make_test_thread([&] { ss.request_stop(); });

    startedIndex.wait(0);

    // now one of the callback has started but not finished.
    if (startedIndex == 1) {
      sc2.reset();   // destructor should not block
    } else if (startedIndex == 2) {
      sc1.reset();   // destructor should not block
    } else {
      assert(false); // something is wrong
    }

    callbackFinish = true;
    callbackFinish.notify_all();
    thread.join();
  }

  // If callback is concurrently executing on another thread, then the
  // return from the invocation of callback strongly happens before ([intro.races])
  // callback is destroyed.
  {
    struct Callback {
      std::atomic<bool>& started_;
      std::atomic<bool>& waitDone_;
      std::atomic<bool>& finished_;
      bool moved = false;

      Callback(std::atomic<bool>& started, std::atomic<bool>& waitDone, std::atomic<bool>& finished)
          : started_(started), waitDone_(waitDone), finished_(finished) {}
      Callback(Callback&& other) : started_(other.started_), waitDone_(other.waitDone_), finished_(other.finished_) {
        other.moved = true;
      }

      void operator()() const {
        struct ScopedGuard {
          std::atomic<bool>& g_finished_;
          ~ScopedGuard() { g_finished_.store(true, std::memory_order_relaxed); }
        };

        started_ = true;
        started_.notify_all();
        waitDone_.wait(false);
        ScopedGuard g{finished_};
      }

      ~Callback() {
        if (!moved) {
          // destructor has to be called after operator() returns
          assert(finished_.load(std::memory_order_relaxed));
        }
      }
    };

    std::stop_source ss;

    std::atomic<bool> started  = false;
    std::atomic<bool> waitDone = false;
    std::atomic<bool> finished = false;

    std::optional<std::stop_callback<Callback>> sc{
        std::in_place, ss.get_token(), Callback{started, waitDone, finished}};

    auto thread1 = support::make_test_thread([&] { ss.request_stop(); });
    started.wait(false);

    auto thread2 = support::make_test_thread([&] {
      using namespace std::chrono_literals;
      std::this_thread::sleep_for(1ms);
      waitDone = true;
      waitDone.notify_all();
    });

    sc.reset(); // destructor should block until operator() returns, i.e. waitDone to be true

    thread1.join();
    thread2.join();
  }

  // If callback is executing on the current thread, then the destructor does not block ([defns.block])
  // waiting for the return from the invocation of callback.
  {
    std::stop_source ss;

    CallbackHolder holder;
    holder.callback_ = std::make_unique<std::stop_callback<DeleteHolder>>(ss.get_token(), DeleteHolder{holder});

    assert(holder.callback_ != nullptr);

    ss.request_stop(); // the callbacks deletes itself. if the destructor blocks, it would be deadlock
    assert(holder.callback_ == nullptr);
  }

  return 0;
}

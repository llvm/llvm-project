// RUN: %clang_analyze_cc1 -std=c++20 -analyzer-checker=core -verify %s

// Regression test for the Clang Static Analyzer treating a coroutine
// promise object's data members as uninitialized garbage even when they
// are guaranteed to be initialized (via in-class default member
// initializers, a user-provided constructor, brace-init, etc.), because
// the promise object's construction was never represented in the CFG that
// the analyzer walks. See the CFG.cpp change to
// CFG::BuildOptions::AddImplicitCoroutinePromiseConstruction.

#include "Inputs/system-header-simulator-cxx-coroutines.h"

struct Ready {
  bool await_ready() noexcept { return true; }
  void await_suspend(std::coroutine_handle<>) noexcept {}
  void await_resume() noexcept {}
};

//===----------------------------------------------------------------------===//
// 1. Exact reproducer from the bug report: in-class default member
//    initializer on a scalar member. The original report used
//    std::uintptr_t (as in real pointer-tagging code), but the exact
//    integer type is irrelevant to the bug, so a plain, unambiguously
//    portable type is used here instead.
//===----------------------------------------------------------------------===//

struct PointerTask {
  struct promise_type {
    unsigned storage_ = 0;

    PointerTask get_return_object() noexcept { return {}; }
    std::suspend_never initial_suspend() noexcept { return {}; }
    std::suspend_never final_suspend() noexcept { return {}; }
    void return_void() noexcept {}
    void unhandled_exception() noexcept {}

    Ready await_transform(Ready a) noexcept {
      (void)(storage_ & ~3u); // no-warning
      return a;
    }
  };
};

PointerTask pointer_repro() {
  co_await Ready{};
}

//===----------------------------------------------------------------------===//
// 2. Multiple members, each with its own default member initializer.
//===----------------------------------------------------------------------===//

struct MultiMemberTask {
  struct promise_type {
    int a_ = 1;
    int b_ = 2;
    bool flag_ = false;

    MultiMemberTask get_return_object() noexcept { return {}; }
    std::suspend_never initial_suspend() noexcept { return {}; }
    std::suspend_never final_suspend() noexcept { return {}; }
    void return_void() noexcept {}
    void unhandled_exception() noexcept {}

    Ready await_transform(Ready a) noexcept {
      if (flag_) {}                   // no-warning
      (void)(a_ + b_);                // no-warning
      return a;
    }
  };
};

MultiMemberTask multi_member_repro() {
  co_await Ready{};
}

//===----------------------------------------------------------------------===//
// 3. Member initialized by the promise type's user-provided (non-default)
//    constructor body, rather than an in-class initializer.
//===----------------------------------------------------------------------===//

struct CtorInitTask {
  struct promise_type {
    int value_;

    promise_type() : value_(0) {}

    CtorInitTask get_return_object() noexcept { return {}; }
    std::suspend_never initial_suspend() noexcept { return {}; }
    std::suspend_never final_suspend() noexcept { return {}; }
    void return_void() noexcept {}
    void unhandled_exception() noexcept {}

    Ready await_transform(Ready a) noexcept {
      (void)(value_ & 1); // no-warning
      return a;
    }
  };
};

CtorInitTask ctor_init_repro() {
  co_await Ready{};
}

//===----------------------------------------------------------------------===//
// 4. Brace / zero initialization of a scalar member.
//===----------------------------------------------------------------------===//

struct BraceInitTask {
  struct promise_type {
    int value_{};

    BraceInitTask get_return_object() noexcept { return {}; }
    std::suspend_never initial_suspend() noexcept { return {}; }
    std::suspend_never final_suspend() noexcept { return {}; }
    void return_void() noexcept {}
    void unhandled_exception() noexcept {}

    Ready await_transform(Ready a) noexcept {
      (void)(value_ | 1); // no-warning
      return a;
    }
  };
};

BraceInitTask brace_init_repro() {
  co_await Ready{};
}

//===----------------------------------------------------------------------===//
// 5. Negative control: a promise member that is genuinely never
//    initialized must still be flagged. This guards against the fix
//    degenerating into a blanket suppression for promise types.
//===----------------------------------------------------------------------===//

struct UninitTask {
  struct promise_type {
    int uninitialized_;

    UninitTask get_return_object() noexcept { return {}; }
    std::suspend_never initial_suspend() noexcept { return {}; }
    std::suspend_never final_suspend() noexcept { return {}; }
    void return_void() noexcept {}
    void unhandled_exception() noexcept {}

    Ready await_transform(Ready a) noexcept {
      (void)(uninitialized_ & 1); // expected-warning{{The left operand of '&' is a garbage value}}
      return a;
    }
  };
};

UninitTask uninit_repro() {
  co_await Ready{};
}

// RUN: %check_clang_tidy -std=c++23-or-later %s cppcoreguidelines-avoid-capturing-lambda-coroutines %t \
// RUN:   -- -config='{CheckOptions: {cppcoreguidelines-avoid-capturing-lambda-coroutines.AllowExplicitObjectParameters: true}}' \
// RUN:   -- -isystem %S/Inputs/system

#include <coroutines.h>

// --- Cases that SHOULD still trigger the warning ---

void test_capture_coroutine_no_deducing_this() {
  int x = 42;
  [&x]() -> task {
    // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: coroutine lambda may cause use-after-free, avoid captures or ensure lambda closure object has guaranteed lifetime [cppcoreguidelines-avoid-capturing-lambda-coroutines]
    co_return;
  };
}

void test_capture_coroutine_with_params() {
  int x = 42;
  [&x](int a) -> task {
    // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: coroutine lambda may cause use-after-free
    co_return;
  };
}

void test_default_capture_ref() {
  int x = 42;
  [&]() -> task {
    // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: coroutine lambda may cause use-after-free
    (void)x;
    co_return;
  };
}

void test_default_capture_copy() {
  int x = 42;
  [=]() -> task {
    // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: coroutine lambda may cause use-after-free
    (void)x;
    co_return;
  };
}

struct S {
  void test_this_capture() {
    [this]() -> task {
      // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: coroutine lambda may cause use-after-free
      co_return;
    };
  }
};

// --- Cases that should NOT trigger the warning ---

void test_deducing_this_coroutine() {
  int x = 42;
  [&x](this auto) -> task { co_return; };
}

void test_deducing_this_with_params() {
  int x = 42;
  [&x](this auto, int a) -> task { co_return; };
}

void test_deducing_this_with_template() {
  int x = 42;
  [&x]<typename T>(this auto, T a) -> task { co_return; };
}

void test_deducing_this_ref_qualified() {
  int x = 42;
  [&x](this auto&&) -> task { co_return; };
}

template<typename T>
concept Integral = requires(T t) { t + 1; };

void test_deducing_this_with_requires() {
  int x = 42;
  [&x]<typename T>(this auto, T a) -> task requires Integral<T> { co_return; };
}

void test_no_captures_no_coroutine() {
  []() { return; };
}

void test_no_captures_coroutine() {
  []() -> task { co_return; };
}

void test_captures_not_coroutine() {
  int x = 42;
  [&x]() { (void)x; };
}

// RUN: %check_clang_tidy -std=c++23 %s concurrency-lambda-coroutine-capture %t

// Minimal coroutine support types.
namespace std {

template <typename R, typename...> struct coroutine_traits {
  using promise_type = typename R::promise_type;
};

template <typename Promise = void> struct coroutine_handle;

template <> struct coroutine_handle<void> {
  static coroutine_handle from_address(void *);
  coroutine_handle() = default;
  coroutine_handle(decltype(nullptr)) {}
};

template <typename Promise> struct coroutine_handle : coroutine_handle<> {
  static coroutine_handle from_address(void *);
};

struct suspend_never {
  bool await_ready() noexcept { return true; }
  void await_suspend(coroutine_handle<>) noexcept {}
  void await_resume() noexcept {}
};

} // namespace std

struct task {
  struct promise_type {
    task get_return_object() { return {}; }
    std::suspend_never initial_suspend() { return {}; }
    std::suspend_never final_suspend() noexcept { return {}; }
    void return_void() {}
    void unhandled_exception() {}
  };
};

template <typename F> void invoke(F &&f) {}

// --- Cases that SHOULD trigger the warning ---

void test_capture_no_parens() {
  int x = 42;
  invoke([&x] -> task {
    // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: lambda coroutine with captures may cause use-after-free; use 'this auto' as the first parameter to move captures into the coroutine frame [concurrency-lambda-coroutine-capture]
    // CHECK-FIXES: {{^}}  invoke([&x](this auto) -> task {{{$}}
    co_return;
  });
}

void test_capture_empty_parens() {
  int x = 42;
  invoke([&x]() -> task {
    // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: lambda coroutine with captures may cause use-after-free
    // CHECK-FIXES: {{^}}  invoke([&x](this auto) -> task {{{$}}
    co_return;
  });
}

void test_capture_with_params() {
  int x = 42;
  invoke([&x](int a) -> task {
    // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: lambda coroutine with captures may cause use-after-free
    // CHECK-FIXES: {{^}}  invoke([&x](this auto, int a) -> task {{{$}}
    co_return;
  });
}

void test_capture_with_multiple_params() {
  int x = 42;
  invoke([&x](int a, int b) -> task {
    // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: lambda coroutine with captures may cause use-after-free
    // CHECK-FIXES: {{^}}  invoke([&x](this auto, int a, int b) -> task {{{$}}
    co_return;
  });
}

void test_capture_by_value() {
  int x = 42;
  invoke([x]() -> task {
    // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: lambda coroutine with captures may cause use-after-free
    // CHECK-FIXES: {{^}}  invoke([x](this auto) -> task {{{$}}
    co_return;
  });
}

void test_default_capture_ref() {
  int x = 42;
  invoke([&]() -> task {
    // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: lambda coroutine with captures may cause use-after-free
    // CHECK-FIXES: {{^}}  invoke([&](this auto) -> task {{{$}}
    (void)x;
    co_return;
  });
}

void test_default_capture_copy() {
  int x = 42;
  invoke([=]() -> task {
    // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: lambda coroutine with captures may cause use-after-free
    // CHECK-FIXES: {{^}}  invoke([=](this auto) -> task {{{$}}
    (void)x;
    co_return;
  });
}

// --- Cases that should NOT trigger the warning ---

void test_no_captures_no_coroutine() {
  invoke([]() { return; });
}

void test_no_captures_coroutine() {
  invoke([]() -> task { co_return; });
}

void test_deducing_this_coroutine() {
  int x = 42;
  invoke([&x](this auto) -> task { co_return; });
}

void test_deducing_this_with_params() {
  int x = 42;
  invoke([&x](this auto, int a) -> task { co_return; });
}

void test_captures_not_coroutine() {
  int x = 42;
  invoke([&x]() { (void)x; });
}

// RUN: %check_clang_tidy -std=c++20 %s cppcoreguidelines-avoid-reference-coroutine-parameters %t

// NOLINTBEGIN
namespace std {
  template <typename T, typename... Args>
  struct coroutine_traits {
    using promise_type = typename T::promise_type;
  };
  template <typename T = void>
  struct coroutine_handle;
  template <>
  struct coroutine_handle<void> {
    coroutine_handle() noexcept;
    coroutine_handle(decltype(nullptr)) noexcept;
    static constexpr coroutine_handle from_address(void*);
  };
  template <typename T>
  struct coroutine_handle {
    coroutine_handle() noexcept;
    coroutine_handle(decltype(nullptr)) noexcept;
    static constexpr coroutine_handle from_address(void*);
    operator coroutine_handle<>() const noexcept;
  };
} // namespace std

struct Awaiter {
  bool await_ready() noexcept;
  void await_suspend(std::coroutine_handle<>) noexcept;
  void await_resume() noexcept;
};

struct Coro {
  struct promise_type {
    Awaiter initial_suspend();
    Awaiter final_suspend() noexcept;
    void return_void();
    Coro get_return_object();
    void unhandled_exception();
  };
};
// NOLINTEND

struct Obj {};

Coro no_args() {
  co_return;
}

Coro no_references(int x, int* y, Obj z, const Obj w) {
  co_return;
}

Coro accepts_references(int& x, const int &y) {
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: coroutine parameters should not be references [cppcoreguidelines-avoid-reference-coroutine-parameters]
  // CHECK-MESSAGES: :[[@LINE-2]]:33: warning: coroutine parameters should not be references [cppcoreguidelines-avoid-reference-coroutine-parameters]
  co_return;
}

Coro accepts_references_and_non_references(int& x, int y) {
  // CHECK-MESSAGES: :[[@LINE-1]]:44: warning: coroutine parameters should not be references [cppcoreguidelines-avoid-reference-coroutine-parameters]
  co_return;
}

Coro accepts_references_to_objects(Obj& x) {
  // CHECK-MESSAGES: :[[@LINE-1]]:36: warning: coroutine parameters should not be references [cppcoreguidelines-avoid-reference-coroutine-parameters]
  co_return;
}

Coro non_coro_accepts_references(int& x) {
  if (x);
  return Coro{};
}

void defines_a_lambda() {
  auto NoArgs = [](int x) -> Coro { co_return; };

  auto NoReferences = [](int x) -> Coro { co_return; };

  auto WithReferences = [](int& x) -> Coro { co_return; };
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: coroutine parameters should not be references [cppcoreguidelines-avoid-reference-coroutine-parameters]

  auto WithReferences2 = [](int&) -> Coro { co_return; };
  // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: coroutine parameters should not be references [cppcoreguidelines-avoid-reference-coroutine-parameters]
}

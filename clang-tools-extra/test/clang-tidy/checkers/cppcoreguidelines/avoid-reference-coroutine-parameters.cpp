// RUN: %check_clang_tidy -check-suffixes=DEFAULT -std=c++20-or-later %s cppcoreguidelines-avoid-reference-coroutine-parameters %t
// RUN: %check_clang_tidy -check-suffixes=ALLOWED -std=c++20-or-later %s cppcoreguidelines-avoid-reference-coroutine-parameters %t -- -config="{CheckOptions: {cppcoreguidelines-avoid-reference-coroutine-parameters.AllowedReturnTypes: 'RefSafeCoro'}}"

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

// A coroutine task type whose reference parameters are safe by construction,
// used to exercise the 'AllowedReturnTypes' option.
struct RefSafeCoro {
  struct promise_type {
    Awaiter initial_suspend();
    Awaiter final_suspend() noexcept;
    void return_void();
    RefSafeCoro get_return_object();
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
  // CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:25: warning: coroutine parameters should not be references [cppcoreguidelines-avoid-reference-coroutine-parameters]
  // CHECK-MESSAGES-ALLOWED: :[[@LINE-2]]:25: warning: coroutine parameters should not be references [cppcoreguidelines-avoid-reference-coroutine-parameters]
  // CHECK-MESSAGES-DEFAULT: :[[@LINE-3]]:33: warning: coroutine parameters should not be references [cppcoreguidelines-avoid-reference-coroutine-parameters]
  // CHECK-MESSAGES-ALLOWED: :[[@LINE-4]]:33: warning: coroutine parameters should not be references [cppcoreguidelines-avoid-reference-coroutine-parameters]
  co_return;
}

Coro accepts_references_and_non_references(int& x, int y) {
  // CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:44: warning: coroutine parameters should not be references [cppcoreguidelines-avoid-reference-coroutine-parameters]
  // CHECK-MESSAGES-ALLOWED: :[[@LINE-2]]:44: warning: coroutine parameters should not be references [cppcoreguidelines-avoid-reference-coroutine-parameters]
  co_return;
}

Coro accepts_references_to_objects(Obj& x) {
  // CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:36: warning: coroutine parameters should not be references [cppcoreguidelines-avoid-reference-coroutine-parameters]
  // CHECK-MESSAGES-ALLOWED: :[[@LINE-2]]:36: warning: coroutine parameters should not be references [cppcoreguidelines-avoid-reference-coroutine-parameters]
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
  // CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:28: warning: coroutine parameters should not be references [cppcoreguidelines-avoid-reference-coroutine-parameters]
  // CHECK-MESSAGES-ALLOWED: :[[@LINE-2]]:28: warning: coroutine parameters should not be references [cppcoreguidelines-avoid-reference-coroutine-parameters]

  auto WithReferences2 = [](int&) -> Coro { co_return; };
  // CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:29: warning: coroutine parameters should not be references [cppcoreguidelines-avoid-reference-coroutine-parameters]
  // CHECK-MESSAGES-ALLOWED: :[[@LINE-2]]:29: warning: coroutine parameters should not be references [cppcoreguidelines-avoid-reference-coroutine-parameters]
}

void coroInFunctionWithReference(int&) {
  auto SampleCoro = [](int x) -> Coro { co_return; };
}

Coro lambdaWithReferenceInCoro() {
  auto SampleLambda = [](int& x) {};
  co_return;
}

using MyIntegerRef = int&;
Coro coroWithReferenceBehindTypedef(MyIntegerRef ref) {
// CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:37: warning: coroutine parameters should not be references [cppcoreguidelines-avoid-reference-coroutine-parameters]
// CHECK-MESSAGES-ALLOWED: :[[@LINE-2]]:37: warning: coroutine parameters should not be references [cppcoreguidelines-avoid-reference-coroutine-parameters]
  co_return;
}

// The return type matches 'AllowedReturnTypes' in the second run, so the
// reference parameter is exempt there, but is still flagged by default. This
// confirms the option is type-specific and does not blanket-allow references:
// the 'Coro' coroutines above are still flagged in the same run.
RefSafeCoro allowedReturnType(int& x) {
  // CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:31: warning: coroutine parameters should not be references [cppcoreguidelines-avoid-reference-coroutine-parameters]
  co_return;
}

// The return type is matched canonically, so a type alias to an allowed type is
// also exempt in the second run.
using MyTask = RefSafeCoro;
MyTask allowedReturnTypeAlias(int& x) {
  // CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:31: warning: coroutine parameters should not be references [cppcoreguidelines-avoid-reference-coroutine-parameters]
  co_return;
}

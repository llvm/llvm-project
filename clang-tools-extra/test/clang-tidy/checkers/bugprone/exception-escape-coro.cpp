// RUN: %check_clang_tidy -std=c++20-or-later %s bugprone-exception-escape %t -- \
// RUN:     -- -fexceptions -Wno-error=return-type

namespace std {

template <class Ret, typename... T> struct coroutine_traits {
  using promise_type = typename Ret::promise_type;
};

template <class Promise = void> struct coroutine_handle {
  static coroutine_handle from_address(void *) noexcept;
  static coroutine_handle from_promise(Promise &promise);
  constexpr void *address() const noexcept;
};

template <> struct coroutine_handle<void> {
  template <class PromiseType>
  coroutine_handle(coroutine_handle<PromiseType>) noexcept;
  static coroutine_handle from_address(void *);
  constexpr void *address() const noexcept;
};

struct suspend_always {
  bool await_ready() noexcept { return false; }
  void await_suspend(coroutine_handle<>) noexcept {}
  void await_resume() noexcept {}
};

struct suspend_never {
  bool await_ready() noexcept { return true; }
  void await_suspend(coroutine_handle<>) noexcept {}
  void await_resume() noexcept {}
};

} // namespace std

template <typename Task, typename T, bool ThrowInPromiseConstructor,
          bool ThrowInInitialSuspend, bool ThrowInGetReturnObject,
          bool ThrowInUnhandledException, bool RethrowInUnhandledException>
struct Promise;

template <
    typename T, bool ThrowInTaskConstructor = false,
    bool ThrowInPromiseConstructor = false, bool ThrowInInitialSuspend = false,
    bool ThrowInGetReturnObject = false, bool ThrowInUnhandledException = false,
    bool RethrowInUnhandledException = false>
struct Task {
  using promise_type =
      Promise<Task, T, ThrowInPromiseConstructor, ThrowInInitialSuspend,
              ThrowInGetReturnObject, ThrowInUnhandledException, RethrowInUnhandledException>;

  explicit Task(promise_type &p) {
    if constexpr (ThrowInTaskConstructor) {
      throw 1;
    }

    p.return_val = this;
  }

  bool await_ready() { return true; }

  void await_suspend(std::coroutine_handle<> h) {}

  void await_resume() {}

  T value;
};

template <bool ThrowInTaskConstructor, bool ThrowInPromiseConstructor,
          bool ThrowInInitialSuspend, bool ThrowInGetReturnObject,
          bool ThrowInUnhandledException, bool RethrowInUnhandledException>
struct Task<void, ThrowInTaskConstructor, ThrowInPromiseConstructor,
            ThrowInInitialSuspend, ThrowInGetReturnObject,
            ThrowInUnhandledException, RethrowInUnhandledException> {
  using promise_type =
      Promise<Task, void, ThrowInPromiseConstructor, ThrowInInitialSuspend,
              ThrowInGetReturnObject, ThrowInUnhandledException, RethrowInUnhandledException>;

  explicit Task(promise_type &p) {
    if constexpr (ThrowInTaskConstructor) {
      throw 1;
    }

    p.return_val = this;
  }

  bool await_ready() { return true; }

  void await_suspend(std::coroutine_handle<> h) {}

  void await_resume() {}
};

template <typename Task, typename T, bool ThrowInPromiseConstructor,
          bool ThrowInInitialSuspend, bool ThrowInGetReturnObject,
          bool ThrowInUnhandledException, bool RethrowInUnhandledException>
struct Promise {
  Promise() {
    if constexpr (ThrowInPromiseConstructor) {
      throw 1;
    }
  }

  Task get_return_object() {
    if constexpr (ThrowInGetReturnObject) {
      throw 1;
    }

    return Task{*this};
  }

  std::suspend_never initial_suspend() const {
    if constexpr (ThrowInInitialSuspend) {
      throw 1;
    }

    return {};
  }

  std::suspend_never final_suspend() const noexcept { return {}; }

  template <typename U> void return_value(U &&val) {
    return_val->value = static_cast<U &&>(val);
  }

  template <typename U> std::suspend_never yield_value(U &&val) {
    return_val->value = static_cast<U &&>(val);
    return {};
  }

  void unhandled_exception() {
    if constexpr (ThrowInUnhandledException) {
      throw 1;
    } else if constexpr (RethrowInUnhandledException) {
      throw;
    }
  }

  Task *return_val;
};

template <typename Task, bool ThrowInPromiseConstructor,
          bool ThrowInInitialSuspend, bool ThrowInGetReturnObject,
          bool ThrowInUnhandledException, bool RethrowInUnhandledException>
struct Promise<Task, void, ThrowInPromiseConstructor, ThrowInInitialSuspend,
               ThrowInGetReturnObject, ThrowInUnhandledException, RethrowInUnhandledException> {
  Promise() {
    if constexpr (ThrowInPromiseConstructor) {
      throw 1;
    }
  }

  Task get_return_object() {
    if constexpr (ThrowInGetReturnObject) {
      throw 1;
    }

    return Task{*this};
  }

  std::suspend_never initial_suspend() const {
    if constexpr (ThrowInInitialSuspend) {
      throw 1;
    }

    return {};
  }

  std::suspend_never final_suspend() const noexcept { return {}; }

  void return_void() {}

  void unhandled_exception() {
    if constexpr (ThrowInUnhandledException) {
      throw 1;
    } else if constexpr (RethrowInUnhandledException) {
      throw;
    }
  }

  Task *return_val;
};

struct Evil {
  ~Evil() noexcept(false) {
    throw 42;
  }
};

Task<int> returnOne() { co_return 1; }

namespace function {

namespace coreturn {

Task<int> a_ShouldNotDiag(const int a, const int b) {
  if (b == 0)
    throw b;

  co_return a / b;
}

Task<int> b_ShouldNotDiag(const int a, const int b) noexcept {
  if (b == 0)
    throw b;

  co_return a / b;
}

Task<int> c_ShouldNotDiag(const int a, const int b) {
  if (b == 0)
    throw Evil{};

  co_return a / b;
}

Task<int> c_ShouldDiag(const int a, const int b) noexcept {
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: an exception may be thrown in function 'c_ShouldDiag' which should not throw exceptions
  if (b == 0)
    throw Evil{};

  co_return a / b;
}
// CHECK-MESSAGES: :186:5: note: frame #0: unhandled exception of type 'int' may be thrown in function '~Evil' here
// CHECK-MESSAGES: :[[@LINE-8]]:11: note: frame #1: function 'c_ShouldDiag' calls function '~Evil'

Task<int, true> d_ShouldNotDiag(const int a, const int b) {
  co_return a / b;
}

Task<int, true> d_ShouldDiag(const int a, const int b) noexcept {
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: an exception may be thrown in function 'd_ShouldDiag' which should not throw exceptions
  co_return a / b;
}
// CHECK-MESSAGES: :54:7: note: frame #0: unhandled exception of type 'int' may be thrown in function 'Task' here
// CHECK-MESSAGES: :109:12: note: frame #1: function 'get_return_object' calls function 'Task' here
// CHECK-MESSAGES: :[[@LINE-6]]:17: note: frame #2: function 'd_ShouldDiag' calls function 'get_return_object' here

Task<int, false, true> e_ShouldNotDiag(const int a, const int b) {
  co_return a / b;
}

Task<int, false, true> e_ShouldDiag(const int a, const int b) noexcept {
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: an exception may be thrown in function 'e_ShouldDiag' which should not throw exceptions
  co_return a / b;
}
// CHECK-MESSAGES: :100:7: note: frame #0: unhandled exception of type 'int' may be thrown in function 'Promise' here
// CHECK-MESSAGES: :[[@LINE-5]]:24: note: frame #1: function 'e_ShouldDiag' calls function 'Promise' here

Task<int, false, false, true> f_ShouldNotDiag(const int a, const int b) {
  co_return a / b;
}

Task<int, false, false, true> f_ShouldDiag(const int a, const int b) noexcept {
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: an exception may be thrown in function 'f_ShouldDiag' which should not throw exceptions
  co_return a / b;
}
// CHECK-MESSAGES: :114:7: note: frame #0: unhandled exception of type 'int' may be thrown in function 'initial_suspend' here
// CHECK-MESSAGES: :[[@LINE-5]]:31: note: frame #1: function 'f_ShouldDiag' calls function 'initial_suspend' here

Task<int, false, false, false, true> g_ShouldNotDiag(const int a, const int b) {
  co_return a / b;
}

Task<int, false, false, false, true> g_ShouldDiag(const int a,
                                                  const int b) noexcept {
  // CHECK-MESSAGES: :[[@LINE-2]]:38: warning: an exception may be thrown in function 'g_ShouldDiag' which should not throw exceptions
  co_return a / b;
}
// CHECK-MESSAGES: :106:7: note: frame #0: unhandled exception of type 'int' may be thrown in function 'get_return_object' here
// CHECK-MESSAGES: :[[@LINE-6]]:38: note: frame #1: function 'g_ShouldDiag' calls function 'get_return_object' here

Task<int, false, false, false, false, true> h_ShouldNotDiag(const int a,
                                                            const int b) {
  co_return a / b;
}

Task<int, false, false, false, false, true> h_ShouldDiag(const int a,
                                                         const int b) noexcept {
  // CHECK-MESSAGES: :[[@LINE-2]]:45: warning: an exception may be thrown in function 'h_ShouldDiag' which should not throw exceptions
  co_return a / b;
}
// CHECK-MESSAGES: :133:7: note: frame #0: unhandled exception of type 'int' may be thrown in function 'unhandled_exception' here
// CHECK-MESSAGES: :[[@LINE-6]]:45: note: frame #1: function 'h_ShouldDiag' calls function 'unhandled_exception' here

Task<int, false, false, false, false, false, true>
i_ShouldNotDiag(const int a, const int b) {
  co_return a / b;
}

Task<int, false, false, false, false, false, true>
i_ShouldNotDiagNoexcept(const int a, const int b) noexcept {
  co_return a / b;
}

Task<int, false, false, false, false, false, true>
j_ShouldNotDiag(const int a, const int b) {
  if (b == 0)
    throw b;

  co_return a / b;
}

Task<int, false, false, false, false, false, true>
j_ShouldDiag(const int a, const int b) noexcept {
  // CHECK-MESSAGES: :[[@LINE-1]]:1: warning: an exception may be thrown in function 'j_ShouldDiag' which should not throw exceptions
  if (b == 0)
    throw b;

  co_return a / b;
}
// CHECK-MESSAGES: :[[@LINE-4]]:5: note: frame #0: unhandled exception of type 'int' may be thrown in function 'j_ShouldDiag' here

} // namespace coreturn

namespace coyield {

Task<int> a_ShouldNotDiag(const int a, const int b) {
  if (b == 0)
    throw b;

  co_yield a / b;
}

Task<int> b_ShouldNotDiag(const int a, const int b) noexcept {
  if (b == 0)
    throw b;

  co_yield a / b;
}

Task<int> c_ShouldNotDiag(const int a, const int b) {
  if (b == 0)
    throw Evil{};

  co_yield a / b;
}

Task<int> c_ShouldDiag(const int a, const int b) noexcept {
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: an exception may be thrown in function 'c_ShouldDiag' which should not throw exceptions
  if (b == 0)
    throw Evil{};

  co_yield a / b;
}
// CHECK-MESSAGES: :186:5: note: frame #0: unhandled exception of type 'int' may be thrown in function '~Evil' here
// CHECK-MESSAGES: :[[@LINE-8]]:11: note: frame #1: function 'c_ShouldDiag' calls function '~Evil'

Task<int, true> d_ShouldNotDiag(const int a, const int b) {
  co_yield a / b;
}

Task<int, true> d_ShouldDiag(const int a, const int b) noexcept {
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: an exception may be thrown in function 'd_ShouldDiag' which should not throw exceptions
  co_yield a / b;
}
// CHECK-MESSAGES: :54:7: note: frame #0: unhandled exception of type 'int' may be thrown in function 'Task' here
// CHECK-MESSAGES: :109:12: note: frame #1: function 'get_return_object' calls function 'Task' here
// CHECK-MESSAGES: :[[@LINE-6]]:17: note: frame #2: function 'd_ShouldDiag' calls function 'get_return_object' here

Task<int, false, true> e_ShouldNotDiag(const int a, const int b) {
  co_yield a / b;
}

Task<int, false, true> e_ShouldDiag(const int a, const int b) noexcept {
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: an exception may be thrown in function 'e_ShouldDiag' which should not throw exceptions
  co_yield a / b;
}
// CHECK-MESSAGES: :100:7: note: frame #0: unhandled exception of type 'int' may be thrown in function 'Promise' here
// CHECK-MESSAGES: :[[@LINE-5]]:24: note: frame #1: function 'e_ShouldDiag' calls function 'Promise' here

Task<int, false, false, true> f_ShouldNotDiag(const int a, const int b) {
  co_yield a / b;
}

Task<int, false, false, true> f_ShouldDiag(const int a, const int b) noexcept {
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: an exception may be thrown in function 'f_ShouldDiag' which should not throw exceptions
  co_yield a / b;
}
// CHECK-MESSAGES: :114:7: note: frame #0: unhandled exception of type 'int' may be thrown in function 'initial_suspend' here
// CHECK-MESSAGES: :[[@LINE-5]]:31: note: frame #1: function 'f_ShouldDiag' calls function 'initial_suspend' here

Task<int, false, false, false, true> g_ShouldNotDiag(const int a, const int b) {
  co_yield a / b;
}

Task<int, false, false, false, true> g_ShouldDiag(const int a,
                                                  const int b) noexcept {
  // CHECK-MESSAGES: :[[@LINE-2]]:38: warning: an exception may be thrown in function 'g_ShouldDiag' which should not throw exceptions
  co_yield a / b;
}
// CHECK-MESSAGES: :106:7: note: frame #0: unhandled exception of type 'int' may be thrown in function 'get_return_object' here
// CHECK-MESSAGES: :[[@LINE-6]]:38: note: frame #1: function 'g_ShouldDiag' calls function 'get_return_object' here

Task<int, false, false, false, false, true> h_ShouldNotDiag(const int a,
                                                            const int b) {
  co_yield a / b;
}

Task<int, false, false, false, false, true> h_ShouldDiag(const int a,
                                                         const int b) noexcept {
  // CHECK-MESSAGES: :[[@LINE-2]]:45: warning: an exception may be thrown in function 'h_ShouldDiag' which should not throw exceptions
  co_yield a / b;
}
// CHECK-MESSAGES: :133:7: note: frame #0: unhandled exception of type 'int' may be thrown in function 'unhandled_exception' here
// CHECK-MESSAGES: :[[@LINE-6]]:45: note: frame #1: function 'h_ShouldDiag' calls function 'unhandled_exception' here

Task<int, false, false, false, false, false, true>
i_ShouldNotDiag(const int a, const int b) {
  co_yield a / b;
}

Task<int, false, false, false, false, false, true>
i_ShouldNotDiagNoexcept(const int a, const int b) noexcept {
  co_yield a / b;
}

Task<int, false, false, false, false, false, true>
j_ShouldNotDiag(const int a, const int b) {
  if (b == 0)
    throw b;

  co_yield a / b;
}

Task<int, false, false, false, false, false, true>
j_ShouldDiag(const int a, const int b) noexcept {
  // CHECK-MESSAGES: :[[@LINE-1]]:1: warning: an exception may be thrown in function 'j_ShouldDiag' which should not throw exceptions
  if (b == 0)
    throw b;

  co_yield a / b;
}
// CHECK-MESSAGES: :[[@LINE-4]]:5: note: frame #0: unhandled exception of type 'int' may be thrown in function 'j_ShouldDiag' here

} // namespace coyield

namespace coawait {

Task<void> a_ShouldNotDiag(const int a, const int b) {
  if (b == 0)
    throw b;

  co_await returnOne();
}

Task<void> b_ShouldNotDiag(const int a, const int b) noexcept {
  if (b == 0)
    throw b;

  co_await returnOne();
}

Task<void> c_ShouldNotDiag(const int a, const int b) {
  if (b == 0)
    throw Evil{};

  co_await returnOne();
}

Task<void> c_ShouldDiag(const int a, const int b) noexcept {
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: an exception may be thrown in function 'c_ShouldDiag' which should not throw exceptions
  if (b == 0)
    throw Evil{};

  co_await returnOne();
}
// CHECK-MESSAGES: :186:5: note: frame #0: unhandled exception of type 'int' may be thrown in function '~Evil' here
// CHECK-MESSAGES: :[[@LINE-8]]:12: note: frame #1: function 'c_ShouldDiag' calls function '~Evil'

Task<void, true> d_ShouldNotDiag(const int a, const int b) {
  co_await returnOne();
}

Task<void, true> d_ShouldDiag(const int a, const int b) noexcept {
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: an exception may be thrown in function 'd_ShouldDiag' which should not throw exceptions
  co_await returnOne();
}
// CHECK-MESSAGES: :81:7: note: frame #0: unhandled exception of type 'int' may be thrown in function 'Task' here
// CHECK-MESSAGES: :158:12: note: frame #1: function 'get_return_object' calls function 'Task' here
// CHECK-MESSAGES: :[[@LINE-6]]:18: note: frame #2: function 'd_ShouldDiag' calls function 'get_return_object' here

Task<void, false, true> e_ShouldNotDiag(const int a, const int b) {
  co_await returnOne();
}

Task<void, false, true> e_ShouldDiag(const int a, const int b) noexcept {
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: an exception may be thrown in function 'e_ShouldDiag' which should not throw exceptions
  co_await returnOne();
}
// CHECK-MESSAGES: :149:7: note: frame #0: unhandled exception of type 'int' may be thrown in function 'Promise' here
// CHECK-MESSAGES: :[[@LINE-5]]:25: note: frame #1: function 'e_ShouldDiag' calls function 'Promise' here

Task<void, false, false, true> f_ShouldNotDiag(const int a, const int b) {
  co_await returnOne();
}

Task<void, false, false, true> f_ShouldDiag(const int a, const int b) noexcept {
  // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: an exception may be thrown in function 'f_ShouldDiag' which should not throw exceptions
  co_await returnOne();
}
// CHECK-MESSAGES: :163:7: note: frame #0: unhandled exception of type 'int' may be thrown in function 'initial_suspend' here
// CHECK-MESSAGES: :[[@LINE-5]]:32: note: frame #1: function 'f_ShouldDiag' calls function 'initial_suspend' here

Task<void, false, false, false, true> g_ShouldNotDiag(const int a,
                                                      const int b) {
  co_await returnOne();
}

Task<void, false, false, false, true> g_ShouldDiag(const int a,
                                                   const int b) noexcept {
  // CHECK-MESSAGES: :[[@LINE-2]]:39: warning: an exception may be thrown in function 'g_ShouldDiag' which should not throw exceptions
  co_await returnOne();
}
// CHECK-MESSAGES: :155:7: note: frame #0: unhandled exception of type 'int' may be thrown in function 'get_return_object' here
// CHECK-MESSAGES: :[[@LINE-6]]:39: note: frame #1: function 'g_ShouldDiag' calls function 'get_return_object' here

Task<void, false, false, false, false, true> h_ShouldNotDiag(const int a,
                                                             const int b) {
  co_await returnOne();
}

Task<void, false, false, false, false, true>
h_ShouldDiag(const int a, const int b) noexcept {
  // CHECK-MESSAGES: :[[@LINE-1]]:1: warning: an exception may be thrown in function 'h_ShouldDiag' which should not throw exceptions
  co_await returnOne();
}
// CHECK-MESSAGES: :175:7: note: frame #0: unhandled exception of type 'int' may be thrown in function 'unhandled_exception' here
// CHECK-MESSAGES: :[[@LINE-5]]:1: note: frame #1: function 'h_ShouldDiag' calls function 'unhandled_exception' here

Task<int, false, false, false, false, false, true>
i_ShouldNotDiag(const int a, const int b) {
  co_await returnOne();
}

Task<int, false, false, false, false, false, true>
i_ShouldNotDiagNoexcept(const int a, const int b) noexcept {
  co_await returnOne();
}

Task<int, false, false, false, false, false, true>
j_ShouldNotDiag(const int a, const int b) {
  co_await returnOne();
  if (b == 0)
    throw b;
}

Task<int, false, false, false, false, false, true>
j_ShouldDiag(const int a, const int b) noexcept {
  // CHECK-MESSAGES: :[[@LINE-1]]:1: warning: an exception may be thrown in function 'j_ShouldDiag' which should not throw exceptions
  co_await returnOne();
  if (b == 0)
    throw b;
}
// CHECK-MESSAGES: :[[@LINE-2]]:5: note: frame #0: unhandled exception of type 'int' may be thrown in function 'j_ShouldDiag' here

} // namespace coawait

} // namespace function

namespace lambda {

namespace coreturn {

const auto a_ShouldNotDiag = [](const int a, const int b) -> Task<int> {
  if (b == 0)
    throw b;

  co_return a / b;
};

const auto b_ShouldNotDiag = [](const int a,
                                const int b) noexcept -> Task<int> {
  if (b == 0)
    throw b;

  co_return a / b;
};

const auto c_ShouldNotDiag = [](const int a, const int b) -> Task<int> {
  if (b == 0)
    throw Evil{};

  co_return a / b;
};

const auto c_ShouldDiag = [](const int a, const int b) noexcept -> Task<int> {
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: an exception may be thrown in function 'operator()' which should not throw exceptions
  if (b == 0)
    throw Evil{};

  co_return a / b;
};
// CHECK-MESSAGES: :186:5: note: frame #0: unhandled exception of type 'int' may be thrown in function '~Evil' here
// CHECK-MESSAGES: :[[@LINE-8]]:27: note: frame #1: function 'operator()' calls function '~Evil'

const auto d_ShouldNotDiag = [](const int a, const int b) -> Task<int, true> {
  co_return a / b;
};

const auto d_ShouldDiag = [](const int a,
                             const int b) noexcept -> Task<int, true> {
  // CHECK-MESSAGES: :[[@LINE-2]]:27: warning: an exception may be thrown in function 'operator()' which should not throw exceptions
  co_return a / b;
};
// CHECK-MESSAGES: :54:7: note: frame #0: unhandled exception of type 'int' may be thrown in function 'Task' here
// CHECK-MESSAGES: :109:12: note: frame #1: function 'get_return_object' calls function 'Task' here
// CHECK-MESSAGES: :[[@LINE-7]]:27: note: frame #2: function 'operator()' calls function 'get_return_object' here

const auto e_ShouldNotDiag = [](const int a,
                                const int b) -> Task<int, false, true> {
  co_return a / b;
};

const auto e_ShouldDiag = [](const int a,
                             const int b) noexcept -> Task<int, false, true> {
  // CHECK-MESSAGES: :[[@LINE-2]]:27: warning: an exception may be thrown in function 'operator()' which should not throw exceptions
  co_return a / b;
};
// CHECK-MESSAGES: :100:7: note: frame #0: unhandled exception of type 'int' may be thrown in function 'Promise' here
// CHECK-MESSAGES: :[[@LINE-6]]:27: note: frame #1: function 'operator()' calls function 'Promise' here

const auto f_ShouldNotDiag = [](const int a,
                                const int b) -> Task<int, false, false, true> {
  co_return a / b;
};

const auto f_ShouldDiag =
    [](const int a, const int b) noexcept -> Task<int, false, false, true> {
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: an exception may be thrown in function 'operator()' which should not throw exceptions
  co_return a / b;
};
// CHECK-MESSAGES: :114:7: note: frame #0: unhandled exception of type 'int' may be thrown in function 'initial_suspend' here
// CHECK-MESSAGES: :[[@LINE-5]]:5: note: frame #1: function 'operator()' calls function 'initial_suspend' here

const auto g_ShouldNotDiag =
    [](const int a, const int b) -> Task<int, false, false, false, true> {
  co_return a / b;
};

const auto g_ShouldDiag =
    [](const int a,
       const int b) noexcept -> Task<int, false, false, false, true> {
  // CHECK-MESSAGES: :[[@LINE-2]]:5: warning: an exception may be thrown in function 'operator()' which should not throw exceptions
  co_return a / b;
};
// CHECK-MESSAGES: :106:7: note: frame #0: unhandled exception of type 'int' may be thrown in function 'get_return_object' here
// CHECK-MESSAGES: :[[@LINE-6]]:5: note: frame #1: function 'operator()' calls function 'get_return_object' here

const auto h_ShouldNotDiag =
    [](const int a,
       const int b) -> Task<int, false, false, false, false, true> {
  co_return a / b;
};

const auto h_ShouldDiag =
    [](const int a,
       const int b) noexcept -> Task<int, false, false, false, false, true> {
  // CHECK-MESSAGES: :[[@LINE-2]]:5: warning: an exception may be thrown in function 'operator()' which should not throw exceptions
  co_return a / b;
};
// CHECK-MESSAGES: :133:7: note: frame #0: unhandled exception of type 'int' may be thrown in function 'unhandled_exception' here
// CHECK-MESSAGES: :[[@LINE-6]]:5: note: frame #1: function 'operator()' calls function 'unhandled_exception' here

const auto i_ShouldNotDiag =
    [](const int a,
       const int b) -> Task<int, false, false, false, false, false, true> {
  co_return a / b;
};

const auto i_ShouldNotDiagNoexcept =
    [](const int a,
       const int b) noexcept -> Task<int, false, false, false, false, false, true> {
  co_return a / b;
};

const auto j_ShouldNotDiag =
    [](const int a,
       const int b) -> Task<int, false, false, false, false, false, true> {
  if (b == 0)
    throw b;

  co_return a / b;
};

const auto j_ShouldDiag =
    [](const int a,
       const int b) noexcept -> Task<int, false, false, false, false, false, true> {
  // CHECK-MESSAGES: :[[@LINE-2]]:5: warning: an exception may be thrown in function 'operator()' which should not throw exceptions
  if (b == 0)
    throw b;

  co_return a / b;
};
// CHECK-MESSAGES: :[[@LINE-4]]:5: note: frame #0: unhandled exception of type 'int' may be thrown in function 'operator()' here

} // namespace coreturn

namespace coyield {

const auto a_ShouldNotDiag = [](const int a, const int b) -> Task<int> {
  if (b == 0)
    throw b;

  co_yield a / b;
};

const auto b_ShouldNotDiag = [](const int a,
                                const int b) noexcept -> Task<int> {
  if (b == 0)
    throw b;

  co_yield a / b;
};

const auto c_ShouldNotDiag = [](const int a, const int b) -> Task<int> {
  if (b == 0)
    throw Evil{};

  co_yield a / b;
};

const auto c_ShouldDiag = [](const int a, const int b) noexcept -> Task<int> {
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: an exception may be thrown in function 'operator()' which should not throw exceptions
  if (b == 0)
    throw Evil{};

  co_yield a / b;
};
// CHECK-MESSAGES: :186:5: note: frame #0: unhandled exception of type 'int' may be thrown in function '~Evil' here
// CHECK-MESSAGES: :[[@LINE-8]]:27: note: frame #1: function 'operator()' calls function '~Evil'

const auto d_ShouldNotDiag = [](const int a, const int b) -> Task<int, true> {
  co_yield a / b;
};

const auto d_ShouldDiag = [](const int a,
                             const int b) noexcept -> Task<int, true> {
  // CHECK-MESSAGES: :[[@LINE-2]]:27: warning: an exception may be thrown in function 'operator()' which should not throw exceptions
  co_yield a / b;
};
// CHECK-MESSAGES: :54:7: note: frame #0: unhandled exception of type 'int' may be thrown in function 'Task' here
// CHECK-MESSAGES: :109:12: note: frame #1: function 'get_return_object' calls function 'Task' here
// CHECK-MESSAGES: :[[@LINE-7]]:27: note: frame #2: function 'operator()' calls function 'get_return_object' here

const auto e_ShouldNotDiag = [](const int a,
                                const int b) -> Task<int, false, true> {
  co_yield a / b;
};

const auto e_ShouldDiag = [](const int a,
                             const int b) noexcept -> Task<int, false, true> {
  // CHECK-MESSAGES: :[[@LINE-2]]:27: warning: an exception may be thrown in function 'operator()' which should not throw exceptions
  co_yield a / b;
};
// CHECK-MESSAGES: :100:7: note: frame #0: unhandled exception of type 'int' may be thrown in function 'Promise' here
// CHECK-MESSAGES: :[[@LINE-6]]:27: note: frame #1: function 'operator()' calls function 'Promise' here

const auto f_ShouldNotDiag = [](const int a,
                                const int b) -> Task<int, false, false, true> {
  co_yield a / b;
};

const auto f_ShouldDiag =
    [](const int a, const int b) noexcept -> Task<int, false, false, true> {
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: an exception may be thrown in function 'operator()' which should not throw exceptions
  co_yield a / b;
};
// CHECK-MESSAGES: :114:7: note: frame #0: unhandled exception of type 'int' may be thrown in function 'initial_suspend' here
// CHECK-MESSAGES: :[[@LINE-5]]:5: note: frame #1: function 'operator()' calls function 'initial_suspend' here

const auto g_ShouldNotDiag =
    [](const int a, const int b) -> Task<int, false, false, false, true> {
  co_yield a / b;
};

const auto g_ShouldDiag =
    [](const int a,
       const int b) noexcept -> Task<int, false, false, false, true> {
  // CHECK-MESSAGES: :[[@LINE-2]]:5: warning: an exception may be thrown in function 'operator()' which should not throw exceptions
  co_yield a / b;
};
// CHECK-MESSAGES: :106:7: note: frame #0: unhandled exception of type 'int' may be thrown in function 'get_return_object' here
// CHECK-MESSAGES: :[[@LINE-6]]:5: note: frame #1: function 'operator()' calls function 'get_return_object' here

const auto h_ShouldNotDiag =
    [](const int a,
       const int b) -> Task<int, false, false, false, false, true> {
  co_yield a / b;
};

const auto h_ShouldDiag =
    [](const int a,
       const int b) noexcept -> Task<int, false, false, false, false, true> {
  // CHECK-MESSAGES: :[[@LINE-2]]:5: warning: an exception may be thrown in function 'operator()' which should not throw exceptions
  co_yield a / b;
};
// CHECK-MESSAGES: :133:7: note: frame #0: unhandled exception of type 'int' may be thrown in function 'unhandled_exception' here
// CHECK-MESSAGES: :[[@LINE-6]]:5: note: frame #1: function 'operator()' calls function 'unhandled_exception' here

const auto i_ShouldNotDiag =
    [](const int a,
       const int b) -> Task<int, false, false, false, false, false, true> {
  co_yield a / b;
};

const auto i_ShouldNotDiagNoexcept =
    [](const int a,
       const int b) noexcept -> Task<int, false, false, false, false, false, true> {
  co_yield a / b;
};

const auto j_ShouldNotDiag =
    [](const int a,
       const int b) -> Task<int, false, false, false, false, false, true> {
  if (b == 0)
    throw b;

  co_yield a / b;
};

const auto j_ShouldDiag =
    [](const int a,
       const int b) noexcept -> Task<int, false, false, false, false, false, true> {
  // CHECK-MESSAGES: :[[@LINE-2]]:5: warning: an exception may be thrown in function 'operator()' which should not throw exceptions
  if (b == 0)
    throw b;

  co_yield a / b;
};
// CHECK-MESSAGES: :[[@LINE-4]]:5: note: frame #0: unhandled exception of type 'int' may be thrown in function 'operator()' here

} // namespace coyield

namespace coawait {

const auto a_ShouldNotDiag = [](const int a, const int b) -> Task<void> {
  if (b == 0)
    throw b;

  co_await returnOne();
};

const auto b_ShouldNotDiag = [](const int a,
                                const int b) noexcept -> Task<void> {
  if (b == 0)
    throw b;

  co_await returnOne();
};

const auto c_ShouldNotDiag = [](const int a, const int b) -> Task<void> {
  if (b == 0)
    throw Evil{};

  co_await returnOne();
};

const auto c_ShouldDiag = [](const int a, const int b) noexcept -> Task<void> {
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: an exception may be thrown in function 'operator()' which should not throw exceptions
  if (b == 0)
    throw Evil{};

  co_await returnOne();
};
// CHECK-MESSAGES: :186:5: note: frame #0: unhandled exception of type 'int' may be thrown in function '~Evil' here
// CHECK-MESSAGES: :[[@LINE-8]]:27: note: frame #1: function 'operator()' calls function '~Evil'

const auto d_ShouldNotDiag = [](const int a, const int b) -> Task<void, true> {
  co_await returnOne();
};

const auto d_ShouldDiag = [](const int a,
                             const int b) noexcept -> Task<void, true> {
  // CHECK-MESSAGES: :[[@LINE-2]]:27: warning: an exception may be thrown in function 'operator()' which should not throw exceptions
  co_await returnOne();
};
// CHECK-MESSAGES: :81:7: note: frame #0: unhandled exception of type 'int' may be thrown in function 'Task' here
// CHECK-MESSAGES: :158:12: note: frame #1: function 'get_return_object' calls function 'Task' here
// CHECK-MESSAGES: :[[@LINE-7]]:27: note: frame #2: function 'operator()' calls function 'get_return_object' here

const auto e_ShouldNotDiag = [](const int a,
                                const int b) -> Task<void, false, true> {
  co_await returnOne();
};

const auto e_ShouldDiag = [](const int a,
                             const int b) noexcept -> Task<void, false, true> {
  // CHECK-MESSAGES: :[[@LINE-2]]:27: warning: an exception may be thrown in function 'operator()' which should not throw exceptions
  co_await returnOne();
};
// CHECK-MESSAGES: :149:7: note: frame #0: unhandled exception of type 'int' may be thrown in function 'Promise' here
// CHECK-MESSAGES: :[[@LINE-6]]:27: note: frame #1: function 'operator()' calls function 'Promise' here

const auto f_ShouldNotDiag = [](const int a,
                                const int b) -> Task<void, false, false, true> {
  co_await returnOne();
};

const auto f_ShouldDiag =
    [](const int a, const int b) noexcept -> Task<void, false, false, true> {
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: an exception may be thrown in function 'operator()' which should not throw exceptions
  co_await returnOne();
};
// CHECK-MESSAGES: :163:7: note: frame #0: unhandled exception of type 'int' may be thrown in function 'initial_suspend' here
// CHECK-MESSAGES: :[[@LINE-5]]:5: note: frame #1: function 'operator()' calls function 'initial_suspend' here

const auto g_ShouldNotDiag =
    [](const int a, const int b) -> Task<void, false, false, false, true> {
  co_await returnOne();
};

const auto g_ShouldDiag =
    [](const int a,
       const int b) noexcept -> Task<void, false, false, false, true> {
  // CHECK-MESSAGES: :[[@LINE-2]]:5: warning: an exception may be thrown in function 'operator()' which should not throw exceptions
  co_await returnOne();
};
// CHECK-MESSAGES: :155:7: note: frame #0: unhandled exception of type 'int' may be thrown in function 'get_return_object' here
// CHECK-MESSAGES: :[[@LINE-6]]:5: note: frame #1: function 'operator()' calls function 'get_return_object' here

const auto h_ShouldNotDiag =
    [](const int a,
       const int b) -> Task<void, false, false, false, false, true> {
  co_await returnOne();
};

const auto h_ShouldDiag =
    [](const int a,
       const int b) noexcept -> Task<void, false, false, false, false, true> {
  // CHECK-MESSAGES: :[[@LINE-2]]:5: warning: an exception may be thrown in function 'operator()' which should not throw exceptions
  co_await returnOne();
};
// CHECK-MESSAGES: :175:7: note: frame #0: unhandled exception of type 'int' may be thrown in function 'unhandled_exception' here
// CHECK-MESSAGES: :[[@LINE-6]]:5: note: frame #1: function 'operator()' calls function 'unhandled_exception' here

const auto i_ShouldNotDiag =
    [](const int a,
       const int b) -> Task<int, false, false, false, false, false, true> {
  co_await returnOne();
};

const auto i_ShouldNotDiagNoexcept =
    [](const int a,
       const int b) noexcept -> Task<int, false, false, false, false, false, true> {
  co_await returnOne();
};

const auto j_ShouldNotDiag =
    [](const int a,
       const int b) -> Task<int, false, false, false, false, false, true> {
  co_await returnOne();
  if (b == 0)
    throw b;
};

const auto j_ShouldDiag =
    [](const int a,
       const int b) noexcept -> Task<int, false, false, false, false, false, true> {
  // CHECK-MESSAGES: :[[@LINE-2]]:5: warning: an exception may be thrown in function 'operator()' which should not throw exceptions
  co_await returnOne();
  if (b == 0)
    throw b;
};
// CHECK-MESSAGES: :[[@LINE-2]]:5: note: frame #0: unhandled exception of type 'int' may be thrown in function 'operator()' here

} // namespace coawait

} // namespace lambda

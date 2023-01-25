// RUN: %clang_cc1 -verify -std=c++17 -fcoroutines-ts -fsyntax-only %s

namespace std::experimental {
template <class Promise = void> struct coroutine_handle {
  coroutine_handle() = default;
  static coroutine_handle from_address(void *) noexcept;
};

template <> struct coroutine_handle<void> {
  static coroutine_handle from_address(void *) noexcept;
  coroutine_handle() = default;
  template <class PromiseType>
  coroutine_handle(coroutine_handle<PromiseType>) noexcept;
};

template <class... Args>
struct void_t_imp {
  using type = void;
};
template <class... Args>
using void_t = typename void_t_imp<Args...>::type;

template <class T, class = void>
struct traits_sfinae_base {};

template <class T>
struct traits_sfinae_base<T, void_t<typename T::promise_type>> {
  using promise_type = typename T::promise_type;
};

template <class Ret, class... Args>
struct coroutine_traits : public traits_sfinae_base<Ret> {};
// expected-note@-1{{declared here}}
} // namespace std::experimental

struct suspend_never {
  bool await_ready() noexcept;
  void await_suspend(std::experimental::coroutine_handle<>) noexcept;
  void await_resume() noexcept;
};

struct MoveOnly {
  MoveOnly() = default;
  MoveOnly(const MoveOnly &) = delete;
  MoveOnly(MoveOnly &&) = default;
};

struct NoCopyNoMove {
  NoCopyNoMove() = default;
  NoCopyNoMove(const NoCopyNoMove &) = delete;
};

template <typename T>
struct task {
  struct promise_type {
    auto initial_suspend() { return suspend_never{}; }
    auto final_suspend() noexcept { return suspend_never{}; }
    auto get_return_object() { return task{}; }
    static void unhandled_exception() {}
    void return_value(T &&value) {} // expected-note 4{{passing argument}}
  };
};

task<NoCopyNoMove> local2val() {
  NoCopyNoMove value;
  co_return value; // expected-warning {{support for 'std::experimental::coroutine_traits' will be removed}}
}

task<NoCopyNoMove &> local2ref() {
  NoCopyNoMove value;
  co_return value; // expected-error {{non-const lvalue reference to type 'NoCopyNoMove' cannot bind to a temporary of type 'NoCopyNoMove'}}
}

// We need the move constructor for construction of the coroutine.
task<MoveOnly> param2val(MoveOnly value) {
  co_return value;
}

task<NoCopyNoMove> lvalue2val(NoCopyNoMove &value) {
  co_return value; // expected-error {{rvalue reference to type 'NoCopyNoMove' cannot bind to lvalue of type 'NoCopyNoMove'}}
}

task<NoCopyNoMove> rvalue2val(NoCopyNoMove &&value) {
  co_return value;
}

task<NoCopyNoMove &> lvalue2ref(NoCopyNoMove &value) {
  co_return value;
}

task<NoCopyNoMove &> rvalue2ref(NoCopyNoMove &&value) {
  co_return value; // expected-error {{non-const lvalue reference to type 'NoCopyNoMove' cannot bind to a temporary of type 'NoCopyNoMove'}}
}

struct To {
  operator MoveOnly() &&;
};
task<MoveOnly> conversion_operator() {
  To t;
  co_return t;
}

struct Construct {
  Construct(MoveOnly);
};
task<Construct> converting_constructor() {
  MoveOnly w;
  co_return w;
}

struct Derived : MoveOnly {};
task<MoveOnly> derived2base() {
  Derived result;
  co_return result;
}

struct RetThis {
  task<RetThis> foo() && {
    co_return *this; // expected-error {{rvalue reference to type 'RetThis' cannot bind to lvalue of type 'RetThis'}}
  }
};

template <typename, typename>
struct is_same { static constexpr bool value = false; };

template <typename T>
struct is_same<T, T> { static constexpr bool value = true; };

template <typename T>
struct generic_task {
  struct promise_type {
    auto initial_suspend() { return suspend_never{}; }
    auto final_suspend() noexcept { return suspend_never{}; }
    auto get_return_object() { return generic_task{}; }
    static void unhandled_exception();
    template <typename U>
    void return_value(U &&value) {
      static_assert(is_same<T, U>::value);
    }
  };
};

generic_task<MoveOnly> param2template(MoveOnly value) {
  co_return value; // We should deduce U = MoveOnly.
}

generic_task<NoCopyNoMove &> lvalue2template(NoCopyNoMove &value) {
  co_return value; // We should deduce U = NoCopyNoMove&.
}

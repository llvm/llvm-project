//===-- Integration test for `clang::co_await_suspend_destroy` ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Test for the `coro_await_suspend_destroy` attribute and
// `await_suspend_destroy` method.
//
// Per `AttrDocs.td`, using `coro_await_suspend_destroy` with
// `await_suspend_destroy` should be equivalent to providing a stub
// `await_suspend` that calls `await_suspend_destroy` and then destroys the
// coroutine handle.
//
// This test logs control flow in a variety of scenarios (controlled by
// `test_toggles`), and checks that the execution traces are identical for
// awaiters with/without the attribute. We currently test all combinations of
// error injection points to ensure behavioral equivalence.
//
// In contrast to Clang `lit` tests, this makes it easy to verify non-divergence
// of functional behavior of the entire coroutine across many scenarios,
// including exception handling, early returns, and mixed usage with legacy
// awaitables.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

#if __has_cpp_attribute(clang::coro_await_suspend_destroy)
#  define ATTR_CORO_AWAIT_SUSPEND_DESTROY [[clang::coro_await_suspend_destroy]]
#else
#  define ATTR_CORO_AWAIT_SUSPEND_DESTROY
#endif

#include <cassert>
#include <coroutine>
#include <exception>
#include <iostream>
#include <memory>
#include <optional>
#include <string>

#define DEBUG_LOG 0 // Logs break no-localization CI, set to 1 if needed

#ifndef TEST_HAS_NO_EXCEPTIONS
#  define THROW(_ex) throw _ex;
#else
#  define THROW(_ex)
#endif

struct my_err : std::exception {};

enum test_toggles {
  throw_in_convert_optional_wrapper = 0,
  throw_in_return_value,
  throw_in_await_resume,
  throw_in_await_suspend_destroy,
  dynamic_short_circuit,          // Does not apply to `..._shortcircuits_to_empty` tests
  largest = dynamic_short_circuit // for array in `test_driver`
};

enum test_event {
  unset = 0,
  // Besides events, we also log various integers between 1 and 9999 that
  // disambiguate different awaiters, or represent different return values.
  convert_optional_wrapper = 10000,
  destroy_return_object,
  destroy_promise,
  get_return_object,
  initial_suspend,
  final_suspend,
  return_value,
  throw_return_value,
  unhandled_exception,
  await_ready,
  await_resume,
  destroy_optional_awaitable,
  throw_await_resume,
  await_suspend_destroy,
  throw_await_suspend_destroy,
  await_suspend,
  coro_catch,
  throw_convert_optional_wrapper,
};

struct test_driver {
  static constexpr int max_events = 1000;

  bool toggles_[test_toggles::largest + 1] = {};
  int events_[max_events]                  = {};
  int cur_event_                           = 0;

  bool toggles(test_toggles toggle) const { return toggles_[toggle]; }
  void log(auto&&... events) {
    for (auto event : {static_cast<int>(events)...}) {
      assert(cur_event_ < max_events);
      events_[cur_event_++] = event;
    }
  }
};

// `optional_wrapper` exists since `get_return_object()` can't return
// `std::optional` directly. C++ coroutines have a fundamental timing mismatch
// between when the return object is created and when the value is available:
//
// 1) Early (coroutine startup): `get_return_object()` is called and must return
//    something immediately.
// 2) Later (when `co_return` executes): `return_value(T)` is called with the
//    actual value.
// 3) Issue: If `get_return_object()` returns the storage, it's empty when
//    returned, and writing to it later cannot affect the already-returned copy.
template <typename T>
struct optional_wrapper {
  test_driver& driver_;
  std::optional<T> storage_;
  std::optional<T>*& pointer_;
  optional_wrapper(test_driver& driver, std::optional<T>*& p) : driver_(driver), pointer_(p) { pointer_ = &storage_; }
  operator std::optional<T>() {
    if (driver_.toggles(test_toggles::throw_in_convert_optional_wrapper)) {
      driver_.log(test_event::throw_convert_optional_wrapper);
      THROW(my_err());
    }
    driver_.log(test_event::convert_optional_wrapper);
    return std::move(storage_);
  }
  ~optional_wrapper() { driver_.log(test_event::destroy_return_object); }
};

// Make `std::optional` a coroutine
template <typename T, typename... Args>
struct std::coroutine_traits<std::optional<T>, test_driver&, Args...> {
  struct promise_type {
    std::optional<T>* storagePtr_ = nullptr;
    test_driver& driver_;

    promise_type(test_driver& driver, auto&&...) : driver_(driver) {}
    ~promise_type() { driver_.log(test_event::destroy_promise); }
    optional_wrapper<T> get_return_object() {
      driver_.log(test_event::get_return_object);
      return optional_wrapper<T>(driver_, storagePtr_);
    }
    std::suspend_never initial_suspend() const noexcept {
      driver_.log(test_event::initial_suspend);
      return {};
    }
    std::suspend_never final_suspend() const noexcept {
      driver_.log(test_event::final_suspend);
      return {};
    }
    void return_value(T value) {
      driver_.log(test_event::return_value, value);
      if (driver_.toggles(test_toggles::throw_in_return_value)) {
        driver_.log(test_event::throw_return_value);
        THROW(my_err());
      }
      *storagePtr_ = std::move(value);
    }
    void unhandled_exception() {
      // Leave `*storagePtr_` empty to represent error
      driver_.log(test_event::unhandled_exception);
    }
  };
};

template <typename T, bool HasAttr>
struct base_optional_awaitable {
  test_driver& driver_;
  int id_;
  std::optional<T> opt_;

  ~base_optional_awaitable() { driver_.log(test_event::destroy_optional_awaitable, id_); }

  bool await_ready() const noexcept {
    driver_.log(test_event::await_ready, id_);
    return opt_.has_value();
  }
  T await_resume() {
    if (driver_.toggles(test_toggles::throw_in_await_resume)) {
      driver_.log(test_event::throw_await_resume, id_);
      THROW(my_err());
    }
    driver_.log(test_event::await_resume, id_);
    return std::move(opt_).value();
  }
  void await_suspend_destroy(auto& promise) {
#if __has_cpp_attribute(clang::coro_await_suspend_destroy)
    if constexpr (HasAttr) {
      // This is just here so that old & new events compare exactly equal.
      driver_.log(test_event::await_suspend);
    }
#endif
    assert(promise.storagePtr_);
    if (driver_.toggles(test_toggles::throw_in_await_suspend_destroy)) {
      driver_.log(test_event::throw_await_suspend_destroy, id_);
      THROW(my_err());
    }
    driver_.log(test_event::await_suspend_destroy, id_);
  }
  void await_suspend(auto handle) {
    driver_.log(test_event::await_suspend);
    await_suspend_destroy(handle.promise());
    handle.destroy();
  }
};

template <typename T>
struct old_optional_awaitable : base_optional_awaitable<T, false> {};

template <typename T>
struct ATTR_CORO_AWAIT_SUSPEND_DESTROY new_optional_awaitable : base_optional_awaitable<T, true> {};

void enumerate_toggles(auto lambda) {
  // Generate all combinations of toggle values
  for (int mask = 0; mask <= (1 << (test_toggles::largest + 1)) - 1; ++mask) {
    test_driver driver;
    for (int i = 0; i <= test_toggles::largest; ++i) {
      driver.toggles_[i] = (mask & (1 << i)) != 0;
    }
    lambda(driver);
  }
}

template <typename T>
void check_coro_with_driver_for(auto coro_fn) {
  enumerate_toggles([&](const test_driver& driver) {
    auto old_driver = driver;
    std::optional<T> old_res;
    bool old_threw = false;
#ifndef TEST_HAS_NO_EXCEPTIONS
    try {
#endif
      old_res = coro_fn.template operator()<old_optional_awaitable<T>, T>(old_driver);
#ifndef TEST_HAS_NO_EXCEPTIONS
    } catch (const my_err&) {
      old_threw = true;
    }
#endif
    auto new_driver = driver;
    std::optional<T> new_res;
    bool new_threw = false;
#ifndef TEST_HAS_NO_EXCEPTIONS
    try {
#endif
      new_res = coro_fn.template operator()<new_optional_awaitable<T>, T>(new_driver);
#ifndef TEST_HAS_NO_EXCEPTIONS
    } catch (const my_err&) {
      new_threw = true;
    }
#endif

#if DEBUG_LOG
    // Print toggle values for debugging
    std::string toggle_info = "Toggles: ";
    for (int i = 0; i <= test_toggles::largest; ++i) {
      if (driver.toggles_[i]) {
        toggle_info += std::to_string(i) + " ";
      }
    }
    toggle_info += "\n";
    std::cerr << toggle_info.c_str() << std::endl;
#endif

    assert(old_threw == new_threw);
    assert(old_res == new_res);

    // Compare events arrays directly using cur_event_ and indices
    assert(old_driver.cur_event_ == new_driver.cur_event_);
    for (int i = 0; i < old_driver.cur_event_; ++i) {
      assert(old_driver.events_[i] == new_driver.events_[i]);
    }
  });
}

// Move-only, non-nullable type that quacks like int but stores a
// heap-allocated int. Used to exercise the machinery with a nontrivial type.
class heap_int {
private:
  std::unique_ptr<int> ptr_;

public:
  explicit heap_int(int value) : ptr_(std::make_unique<int>(value)) {}

  heap_int operator+(const heap_int& other) const { return heap_int(*ptr_ + *other.ptr_); }

  bool operator==(const heap_int& other) const { return *ptr_ == *other.ptr_; }

  /*implicit*/ operator int() const { return *ptr_; }
};

void check_coro_with_driver(auto coro_fn) {
  check_coro_with_driver_for<int>(coro_fn);
  check_coro_with_driver_for<heap_int>(coro_fn);
}

template <typename Awaitable, typename T>
std::optional<T> coro_shortcircuits_to_empty(test_driver& driver) {
  T n = co_await Awaitable{driver, 1, std::optional<T>{11}};
  co_await Awaitable{driver, 2, std::optional<T>{}}; // return early!
  co_return n + co_await Awaitable{driver, 3, std::optional<T>{22}};
}

void test_coro_shortcircuits_to_empty() {
#if DEBUG_LOG
  std::cerr << "test_coro_shortcircuits_to_empty" << std::endl;
#endif
  check_coro_with_driver([]<typename Awaitable, typename T>(test_driver& driver) {
    return coro_shortcircuits_to_empty<Awaitable, T>(driver);
  });
}

template <typename Awaitable, typename T>
std::optional<T> coro_simple_await(test_driver& driver) {
  co_return co_await Awaitable{driver, 1, std::optional<T>{11}} +
      co_await Awaitable{driver, 2, driver.toggles(dynamic_short_circuit) ? std::optional<T>{} : std::optional<T>{22}};
}

void test_coro_simple_await() {
#if DEBUG_LOG
  std::cerr << "test_coro_simple_await" << std::endl;
#endif
  check_coro_with_driver([]<typename Awaitable, typename T>(test_driver& driver) {
    return coro_simple_await<Awaitable, T>(driver);
  });
}

// The next pair of tests checks that adding a `try-catch` in the coroutine
// doesn't affect control flow when `await_suspend_destroy` awaiters are in use.

template <typename Awaitable, typename T>
std::optional<T> coro_catching_shortcircuits_to_empty(test_driver& driver) {
#ifndef TEST_HAS_NO_EXCEPTIONS
  try {
#endif
    T n = co_await Awaitable{driver, 1, std::optional<T>{11}};
    co_await Awaitable{driver, 2, std::optional<T>{}}; // return early!
    co_return n + co_await Awaitable{driver, 3, std::optional<T>{22}};
#ifndef TEST_HAS_NO_EXCEPTIONS
  } catch (...) {
    driver.log(test_event::coro_catch);
    throw;
  }
#endif
}

void test_coro_catching_shortcircuits_to_empty() {
#if DEBUG_LOG
  std::cerr << "test_coro_catching_shortcircuits_to_empty" << std::endl;
#endif
  check_coro_with_driver([]<typename Awaitable, typename T>(test_driver& driver) {
    return coro_catching_shortcircuits_to_empty<Awaitable, T>(driver);
  });
}

template <typename Awaitable, typename T>
std::optional<T> coro_catching_simple_await(test_driver& driver) {
#ifndef TEST_HAS_NO_EXCEPTIONS
  try {
#endif
    co_return co_await Awaitable{driver, 1, std::optional<T>{11}} +
        co_await Awaitable{
            driver, 2, driver.toggles(dynamic_short_circuit) ? std::optional<T>{} : std::optional<T>{22}};
#ifndef TEST_HAS_NO_EXCEPTIONS
  } catch (...) {
    driver.log(test_event::coro_catch);
    throw;
  }
#endif
}

void test_coro_catching_simple_await() {
#if DEBUG_LOG
  std::cerr << "test_coro_catching_simple_await" << std::endl;
#endif
  check_coro_with_driver([]<typename Awaitable, typename T>(test_driver& driver) {
    return coro_catching_simple_await<Awaitable, T>(driver);
  });
}

// The next pair of tests shows that the `await_suspend_destroy` code path works
// correctly, even if it's mixed in a coroutine with legacy awaitables.

template <typename Awaitable, typename T>
std::optional<T> noneliding_coro_shortcircuits_to_empty(test_driver& driver) {
  T n  = co_await Awaitable{driver, 1, std::optional<T>{11}};
  T n2 = co_await old_optional_awaitable<T>{driver, 2, std::optional<T>{22}};
  co_await Awaitable{driver, 3, std::optional<T>{}}; // return early!
  co_return n + n2 + co_await Awaitable{driver, 4, std::optional<T>{44}};
}

void test_noneliding_coro_shortcircuits_to_empty() {
#if DEBUG_LOG
  std::cerr << "test_noneliding_coro_shortcircuits_to_empty" << std::endl;
#endif
  check_coro_with_driver([]<typename Awaitable, typename T>(test_driver& driver) {
    return noneliding_coro_shortcircuits_to_empty<Awaitable, T>(driver);
  });
}

template <typename Awaitable, typename T>
std::optional<T> noneliding_coro_simple_await(test_driver& driver) {
  co_return co_await Awaitable{driver, 1, std::optional<T>{11}} +
      co_await Awaitable{driver, 2, driver.toggles(dynamic_short_circuit) ? std::optional<T>{} : std::optional<T>{22}} +
      co_await old_optional_awaitable<T>{driver, 3, std::optional<T>{33}};
}

void test_noneliding_coro_simple_await() {
#if DEBUG_LOG
  std::cerr << "test_noneliding_coro_simple_await" << std::endl;
#endif
  check_coro_with_driver([]<typename Awaitable, typename T>(test_driver& driver) {
    return noneliding_coro_simple_await<Awaitable, T>(driver);
  });
}

// Test nested coroutines (coroutines that await other coroutines)

template <typename Awaitable, typename T>
std::optional<T> inner_coro(test_driver& driver, int base_id) {
  co_return co_await Awaitable{driver, base_id, std::optional<T>{100}} +
      co_await Awaitable{
          driver, base_id + 1, driver.toggles(dynamic_short_circuit) ? std::optional<T>{} : std::optional<T>{200}};
}

template <typename Awaitable, typename T>
std::optional<T> outer_coro(test_driver& driver) {
  T result1 = co_await Awaitable{driver, 1, inner_coro<Awaitable, T>(driver, 10)};
  T result2 = co_await Awaitable{driver, 2, inner_coro<Awaitable, T>(driver, 20)};
  co_return result1 + result2;
}

void test_nested_coroutines() {
#if DEBUG_LOG
  std::cerr << "test_nested_coroutines" << std::endl;
#endif
  check_coro_with_driver([]<typename Awaitable, typename T>(test_driver& driver) {
    return outer_coro<Awaitable, T>(driver);
  });
}

int main(int, char**) {
  test_coro_shortcircuits_to_empty();
  test_coro_simple_await();
  test_coro_catching_shortcircuits_to_empty();
  test_coro_catching_simple_await();
  test_noneliding_coro_shortcircuits_to_empty();
  test_noneliding_coro_simple_await();
  test_nested_coroutines();
  return 0;
}

//===-- Integration test for `clang::co_await_suspend_destroy` ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This file tests the `[[clang::coro_await_suspend_destroy]]` attribute and the
// `await_suspend_destroy()` method. It is ALSO a change detector for any
// deviations from the pre-recorded standard `await_suspend()` control flow.
//
// How to diagnose a failure:
//   - First, rerun a failure with `DEBUG_LOG 1`.
//   - If you see a mismatch between "gold" and "no attr" outputs, this likely
//     means that a coroutine control flow bug was introduced (or, far less
//     likely, fixed).
//   - On the other hand, mismatches between "gold" and "attr" prove that a bug
//     was introduced in the `[[clang::coro_await_suspend_destroy]]`
//     implementation.
//
// Per `AttrDocs.td`, using `[[clang::coro_await_suspend_destroy]]` with
// `await_suspend_destroy()` should be equivalent to providing a stub
// `await_suspend()` that calls `await_suspend_destroy()` and immediately
// destroys the coroutine handle.
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
#include <concepts>
#include <coroutine>
#include <exception>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "test_macros.h" // only for `TEST_HAS_NO_EXCEPTIONS`

// Using `std::cerr` breaks no-localization CI, set to 1 if needed.
#define DEBUG_LOG 0

// If you've deliberately changed the test's order of events, you have to
// re-bootstrap two sets of golden data.
//
// First, make sure `TEST_HAS_NO_EXCEPTIONS` is NOT defined, and:
//   - Temporarily set `BOOTSTRAP_GOLDEN_TEST_OUTPUT` to 1.
//   - Run this binary, redirecting stdout into
//     `coro_await_suspend_destroy.golden-exceptions.h`.
//   - Run clang-format.
//
// Second, temporarily `#define TEST_HAS_NO_EXCEPTIONS 1`, and repeat, saving
// the output to `coro_await_suspend_destroy.golden-no-exceptions.h`.
//
// Finally, undo both of the `#define` changes.
#define BOOTSTRAP_GOLDEN_TEST_OUTPUT 0

#ifndef TEST_HAS_NO_EXCEPTIONS
#  define THROW_IF_EXCEPTIONS_ENABLED(_ex) throw _ex;
#else
#  define THROW_IF_EXCEPTIONS_ENABLED(_ex)
#endif

struct my_err : std::exception {};

enum class test_toggles {
  throw_in_convert_optional_wrapper = 0x01,
  throw_in_return_value             = 0x02,
  throw_in_await_resume             = 0x04,
  throw_in_await_suspend_destroy    = 0x08,
  dynamic_short_circuit             = 0x10,                 // Does not apply to `..._shortcircuits_to_empty` tests
  largest                           = dynamic_short_circuit // for array in `test_driver`
};

enum class test_event {
  unset = 0,
  // Besides events, we also log various integers between 1 and 63 that
  // disambiguate different awaiters, or represent different return values.
  smallest                 = 64,
  convert_optional_wrapper = smallest,
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
  largest = throw_convert_optional_wrapper
};

struct test_driver {
  static constexpr int max_events = 1000;

  int toggles_ = -1;
  bool threw_  = false;
  // -1 is distinct from all events & values, and the empty state.
  // It will show up only on the "with exceptions" & "threw" paths.
  std::optional<int> result_{-1};
  int next_event_         = 0;
  int events_[max_events] = {};

  void assert_equivalent(const test_driver& other) {
    assert(toggles_ == other.toggles_);
    assert(threw_ == other.threw_);
    assert(result_ == other.result_);
    assert(next_event_ == other.next_event_);
    for (int i = 0; i < next_event_; ++i) {
      assert(events_[i] == other.events_[i]);
    }
  }

  static test_driver from_toggle_mask(int toggles) {
    assert(toggles >= 0);
    assert(toggles < ((int)test_toggles::largest << 1));
    return test_driver{.toggles_ = toggles};
  }

  bool toggles(test_toggles toggle) const {
    assert(toggles_ != -1);
    return ((int)toggles_ & (int)toggle) != 0;
  }

  void log(auto&&... events) {
    (
        [this](auto&& event) {
          auto int_event = static_cast<int>(event);
          // Make sure "values" and "events" don't overlap -- this makes it safe
          // to make the event codes shorter.
          if constexpr (std::same_as< test_event, std::remove_reference_t<decltype(event)>>) {
            assert(int_event >= (int)test_event::smallest);
            assert(int_event <= (int)test_event::largest);
          } else {
            assert(int_event > 0);
            assert(int_event < (int)test_event::smallest);
          }
          assert(next_event_ < max_events);
          events_[next_event_++] = int_event;
        }(events),
        ...);
  }

  std::string repr() const {
    std::string result =
        "{\n    .toggles_ = " + std::to_string(toggles_) + //
        ",\n    .threw_ = " + std::to_string(threw_) +     //
        ",\n    .result_ = " + (result_.has_value() ? std::to_string(*result_) : "{}") +
        ",\n    .next_event_ = " + std::to_string(next_event_) + //
        ",\n    .events_ = {";
    for (int i = 0; i < next_event_; ++i) {
      result += std::to_string(events_[i]);
      result += (i < next_event_ - 1) ? ", " : "}";
    }
    result += "}";
    return result;
  }
};

// We test `optional<int>` and `optional<heap_int>`. Record just the value.
std::optional<int> intify_optional(auto&& opt) {
  if (opt.has_value()) {
    return std::optional<int>{static_cast<int>(opt.value())};
  }
  return {};
}

////////////////////////////////////////////////////////////////////////////////
// The coroutine & awaitables under test
////////////////////////////////////////////////////////////////////////////////

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
    driver_.log(test_event::convert_optional_wrapper);
    if (driver_.toggles(test_toggles::throw_in_convert_optional_wrapper)) {
      driver_.log(test_event::throw_convert_optional_wrapper);
      THROW_IF_EXCEPTIONS_ENABLED(my_err());
    }
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
        THROW_IF_EXCEPTIONS_ENABLED(my_err());
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
    driver_.log(test_event::await_resume, id_);
    if (driver_.toggles(test_toggles::throw_in_await_resume)) {
      driver_.log(test_event::throw_await_resume, id_);
      THROW_IF_EXCEPTIONS_ENABLED(my_err());
    }
    return std::move(opt_).value();
  }
  void await_suspend_destroy(auto& promise) {
#if __has_cpp_attribute(clang::coro_await_suspend_destroy)
    // This makes events compare equal, with and without the attribute. However,
    // if the attribute is broken and `await_suspend()` is the entry point, then
    // `test_event::await_suspend` will get appended TWICE, catching the bug.
    if constexpr (HasAttr) {
      driver_.log(test_event::await_suspend);
    }
#endif
    assert(promise.storagePtr_);
    driver_.log(test_event::await_suspend_destroy, id_);
    if (driver_.toggles(test_toggles::throw_in_await_suspend_destroy)) {
      driver_.log(test_event::throw_await_suspend_destroy, id_);
      THROW_IF_EXCEPTIONS_ENABLED(my_err());
    }
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

////////////////////////////////////////////////////////////////////////////////
// Machinery for driving the test coros & comparing their results
////////////////////////////////////////////////////////////////////////////////

// Generate all combinations of toggle values, with empty event traces
std::vector<test_driver> bootstrap_drivers() {
  std::vector<test_driver> drivers;
  for (int mask = 0; mask < ((int)test_toggles::largest << 1); ++mask) {
    drivers.emplace_back(test_driver::from_toggle_mask(mask));
  }
  return drivers;
}

template <typename T, bool PrintsGold = false>
void check_coro_with_driver_for([[maybe_unused]] const char* test_name, auto& expected_drivers, auto coro_fn) {
  for (auto expected_driver : expected_drivers) {
    auto old_driver = test_driver::from_toggle_mask(expected_driver.toggles_);
#ifndef TEST_HAS_NO_EXCEPTIONS
    try {
#endif
      old_driver.result_ = intify_optional(coro_fn.template operator()<old_optional_awaitable<T>, T>(old_driver));
#ifndef TEST_HAS_NO_EXCEPTIONS
    } catch (const my_err&) {
      old_driver.threw_ = true;
    }
#endif

    auto new_driver = test_driver::from_toggle_mask(expected_driver.toggles_);
#ifndef TEST_HAS_NO_EXCEPTIONS
    try {
#endif
      new_driver.result_ = intify_optional(coro_fn.template operator()<new_optional_awaitable<T>, T>(new_driver));
#ifndef TEST_HAS_NO_EXCEPTIONS
    } catch (const my_err&) {
      new_driver.threw_ = true;
    }
#endif

#if BOOTSTRAP_GOLDEN_TEST_OUTPUT
    if constexpr (PrintsGold) {
      std::cout << old_driver.repr() << "," << std::endl;
    }
#endif
#if DEBUG_LOG
    // NB: This omits "has exceptions" and `T`. See "support/type_id.h".
    std::cerr << test_name << " (no attr) = " << old_driver.repr() << std::endl;
    std::cerr << test_name << " (with attr) = " << new_driver.repr() << std::endl;
    std::cerr << "gold = " << expected_driver.repr() << std::endl << std::endl;
#endif

#if !BOOTSTRAP_GOLDEN_TEST_OUTPUT
    // Outside of boostrap, first check normal control flow against the gold
    old_driver.assert_equivalent(expected_driver);
#endif
    // Always check that the attribute matches normal control flow.
    old_driver.assert_equivalent(new_driver);
  };
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

int check_coro_with_driver(const char* test_name, const auto& expected, auto coro_fn) {
#if BOOTSTRAP_GOLDEN_TEST_OUTPUT
  std::cout << "+" << test_name << "(std::vector<test_driver>{" << std::endl;
#endif
  check_coro_with_driver_for<int, true>(test_name, expected, coro_fn);
#if BOOTSTRAP_GOLDEN_TEST_OUTPUT
  std::cout << "})" << std::endl;
#endif
  // The event trace should be the same for `int` and `heap_int`.
  check_coro_with_driver_for<heap_int>(test_name, expected, coro_fn);
  return 1; // main counts tests
}

////////////////////////////////////////////////////////////////////////////////
// Run these 7 tests with different toggles against a header with golden outputs
////////////////////////////////////////////////////////////////////////////////

template <typename Awaitable, typename T>
std::optional<T> coro_shortcircuits_to_empty(test_driver& driver) {
  T n = co_await Awaitable{driver, 1, std::optional<T>{4}};
  co_await Awaitable{driver, 2, std::optional<T>{}}; // return early!
  co_return n + co_await Awaitable{driver, 3, std::optional<T>{8}};
}

int test_coro_shortcircuits_to_empty(const auto& expected) {
  return check_coro_with_driver(
      "test_coro_shortcircuits_to_empty", expected, []<typename Awaitable, typename T>(test_driver& driver) {
        return coro_shortcircuits_to_empty<Awaitable, T>(driver);
      });
}

template <typename Awaitable, typename T>
std::optional<T> coro_simple_await(test_driver& driver) {
  co_return co_await Awaitable{driver, 1, std::optional<T>{4}} +
      co_await Awaitable{
          driver, 2, driver.toggles(test_toggles::dynamic_short_circuit) ? std::optional<T>{} : std::optional<T>{8}};
}

int test_coro_simple_await(const auto& expected) {
  return check_coro_with_driver(
      "test_coro_simple_await", expected, []<typename Awaitable, typename T>(test_driver& driver) {
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
    T n = co_await Awaitable{driver, 1, std::optional<T>{4}};
    co_await Awaitable{driver, 2, std::optional<T>{}}; // return early!
    co_return n + co_await Awaitable{driver, 3, std::optional<T>{8}};
#ifndef TEST_HAS_NO_EXCEPTIONS
  } catch (...) {
    driver.log(test_event::coro_catch);
    throw;
  }
#endif
}

int test_coro_catching_shortcircuits_to_empty(const auto& expected) {
  return check_coro_with_driver(
      "test_coro_catching_shortcircuits_to_empty", expected, []<typename Awaitable, typename T>(test_driver& driver) {
        return coro_catching_shortcircuits_to_empty<Awaitable, T>(driver);
      });
}

template <typename Awaitable, typename T>
std::optional<T> coro_catching_simple_await(test_driver& driver) {
#ifndef TEST_HAS_NO_EXCEPTIONS
  try {
#endif
    co_return co_await Awaitable{driver, 1, std::optional<T>{4}} +
        co_await Awaitable{
            driver, 2, driver.toggles(test_toggles::dynamic_short_circuit) ? std::optional<T>{} : std::optional<T>{8}};
#ifndef TEST_HAS_NO_EXCEPTIONS
  } catch (...) {
    driver.log(test_event::coro_catch);
    throw;
  }
#endif
}

int test_coro_catching_simple_await(const auto& expected) {
  return check_coro_with_driver(
      "test_coro_catching_simple_await", expected, []<typename Awaitable, typename T>(test_driver& driver) {
        return coro_catching_simple_await<Awaitable, T>(driver);
      });
}

// The next pair of tests shows that the `await_suspend_destroy` code path works
// correctly, even if it's mixed in a coroutine with legacy awaitables.

template <typename Awaitable, typename T>
std::optional<T> noneliding_coro_shortcircuits_to_empty(test_driver& driver) {
  T n  = co_await Awaitable{driver, 1, std::optional<T>{8}};
  T n2 = co_await old_optional_awaitable<T>{driver, 2, std::optional<T>{16}};
  co_await Awaitable{driver, 3, std::optional<T>{}}; // return early!
  co_return n + n2 + co_await Awaitable{driver, 4, std::optional<T>{32}};
}

int test_noneliding_coro_shortcircuits_to_empty(const auto& expected) {
  return check_coro_with_driver(
      "test_noneliding_coro_shortcircuits_to_empty", expected, []<typename Awaitable, typename T>(test_driver& driver) {
        return noneliding_coro_shortcircuits_to_empty<Awaitable, T>(driver);
      });
}

template <typename Awaitable, typename T>
std::optional<T> noneliding_coro_simple_await(test_driver& driver) {
  co_return co_await Awaitable{driver, 1, std::optional<T>{4}} +
      co_await Awaitable{
          driver, 2, driver.toggles(test_toggles::dynamic_short_circuit) ? std::optional<T>{} : std::optional<T>{8}} +
      co_await old_optional_awaitable<T>{driver, 3, std::optional<T>{16}};
}

int test_noneliding_coro_simple_await(const auto& expected) {
  return check_coro_with_driver(
      "test_noneliding_coro_simple_await", expected, []<typename Awaitable, typename T>(test_driver& driver) {
        return noneliding_coro_simple_await<Awaitable, T>(driver);
      });
}

// Test nested coroutines (coroutines that await other coroutines)

template <typename Awaitable, typename T>
std::optional<T> inner_coro(test_driver& driver, int base_id) {
  co_return co_await Awaitable{driver, base_id, std::optional<T>{8}} +
      co_await Awaitable{
          driver,
          base_id + 1,
          driver.toggles(test_toggles::dynamic_short_circuit) ? std::optional<T>{} : std::optional<T>{16}};
}

template <typename Awaitable, typename T>
std::optional<T> outer_coro(test_driver& driver) {
  T result1 = co_await Awaitable{driver, 1, inner_coro<Awaitable, T>(driver, 32)};
  T result2 = co_await Awaitable{driver, 2, inner_coro<Awaitable, T>(driver, 32)};
  co_return result1 + result2;
}

int test_nested_coroutines(const auto& expected) {
  return check_coro_with_driver(
      "test_nested_coroutines", expected, []<typename Awaitable, typename T>(test_driver& driver) {
        return outer_coro<Awaitable, T>(driver);
      });
}

int main(int, char**) {
#if BOOTSTRAP_GOLDEN_TEST_OUTPUT
  // This top-of-file comment aims to reduce confusion when bootstrapping.
#  ifdef TEST_HAS_NO_EXCEPTIONS
  std::cout << "// coro_await_suspend_destroy.golden-no-exceptions.h" << std::endl;
#  else
  std::cout << "// coro_await_suspend_destroy.golden-exceptions.h" << std::endl;
#  endif
#endif
  // Consistency check: return 0 if the #include has 7 tests.
  return 7 !=
         (
#if BOOTSTRAP_GOLDEN_TEST_OUTPUT
             test_coro_shortcircuits_to_empty(bootstrap_drivers()) + test_coro_simple_await(bootstrap_drivers()) +
             test_coro_catching_shortcircuits_to_empty(bootstrap_drivers()) +
             test_coro_catching_simple_await(bootstrap_drivers()) +
             test_noneliding_coro_shortcircuits_to_empty(bootstrap_drivers()) +
             test_noneliding_coro_simple_await(bootstrap_drivers()) + test_nested_coroutines(bootstrap_drivers())
#elif defined(TEST_HAS_NO_EXCEPTIONS)
#  include "coro_await_suspend_destroy.golden-no-exceptions.h"
#else
#  include "coro_await_suspend_destroy.golden-exceptions.h"
#endif
         );
}

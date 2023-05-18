// RUN: %clang_cc1 -triple x86_64-apple-darwin9 %s -std=c++20 \
// RUN:    -fsyntax-only -Wignored-qualifiers -Wno-error=return-type -verify \
// RUN:    -fblocks
#include "Inputs/std-coroutine.h"

using namespace std;

template <class Begin>
struct Awaiter {
  bool await_ready();
  void await_suspend(coroutine_handle<>);
  Begin await_resume();
};

template <class Iter> struct BeginTag { BeginTag() = delete; };
template <class Iter> struct IncTag { IncTag() = delete; };

template <class Iter, bool Delete = false>
struct CoawaitTag { CoawaitTag() = delete; };

template <class T>
struct Iter {
  using value_type = T;
  using reference = T &;
  using pointer = T *;

  IncTag<Iter> operator++();
  reference operator*();
  pointer operator->();
};
template <class T> bool operator==(Iter<T>, Iter<T>);
template <class T> bool operator!=(Iter<T>, Iter<T>);

template <class T>
struct Range {
  BeginTag<Iter<T>> begin();
  Iter<T> end();
};

struct MyForLoopArrayAwaiter {
  struct promise_type {
    MyForLoopArrayAwaiter get_return_object() { return {}; }
    void return_void();
    void unhandled_exception();
    suspend_never initial_suspend();
    suspend_never final_suspend() noexcept;
    template <class T>
    Awaiter<T *> await_transform(T *) = delete; // expected-note {{explicitly deleted}}
  };
};
MyForLoopArrayAwaiter g() {
  int arr[10] = {0};
  for co_await(auto i : arr) {} // expected-warning {{'for co_await' belongs to CoroutineTS instead of C++20, which is deprecated}}
  // expected-error@-1 {{call to deleted member function 'await_transform'}}
  // expected-note@-2 {{'await_transform' implicitly required by 'co_await' here}}
}

struct ForLoopAwaiterBadBeginTransform {
  struct promise_type {
    ForLoopAwaiterBadBeginTransform get_return_object();
    void return_void();
    void unhandled_exception();
    suspend_never initial_suspend();
    suspend_never final_suspend() noexcept;

    template <class T>
    Awaiter<T> await_transform(BeginTag<T>) = delete; // expected-note 1+ {{explicitly deleted}}

    template <class T>
    CoawaitTag<T> await_transform(IncTag<T>); // expected-note 1+ {{candidate}}
  };
};
ForLoopAwaiterBadBeginTransform bad_begin() {
  Range<int> R;
  for co_await(auto i : R) {} // expected-warning {{'for co_await' belongs to CoroutineTS instead of C++20, which is deprecated}}
  // expected-error@-1 {{call to deleted member function 'await_transform'}}
  // expected-note@-2 {{'await_transform' implicitly required by 'co_await' here}}
}
template <class Dummy>
ForLoopAwaiterBadBeginTransform bad_begin_template(Dummy) {
  Range<Dummy> R;
  for co_await(auto i : R) {} // expected-warning {{'for co_await' belongs to CoroutineTS instead of C++20, which is deprecated}}
  // expected-error@-1 {{call to deleted member function 'await_transform'}}
  // expected-note@-2 {{'await_transform' implicitly required by 'co_await' here}}
}
template ForLoopAwaiterBadBeginTransform bad_begin_template(int); // expected-note {{requested here}}

template <class Iter>
Awaiter<Iter> operator co_await(CoawaitTag<Iter, true>) = delete;
// expected-note@-1 1+ {{explicitly deleted}}

struct ForLoopAwaiterBadIncTransform {
  struct promise_type {
    ForLoopAwaiterBadIncTransform get_return_object();
    void return_void();
    void unhandled_exception();
    suspend_never initial_suspend();
    suspend_never final_suspend() noexcept;

    template <class T>
    Awaiter<T> await_transform(BeginTag<T> e);

    template <class T>
    CoawaitTag<T, true> await_transform(IncTag<T>);
  };
};
ForLoopAwaiterBadIncTransform bad_inc_transform() {
  Range<float> R;
  for co_await(auto i : R) {} // expected-warning {{'for co_await' belongs to CoroutineTS instead of C++20, which is deprecated}}
  // expected-error@-1 {{overload resolution selected deleted operator 'co_await'}}
  // expected-note@-2 {{in implicit call to 'operator++' for iterator of type 'Range<float>'}}
}

template <class Dummy>
ForLoopAwaiterBadIncTransform bad_inc_transform_template(Dummy) {
  Range<Dummy> R;
  for co_await(auto i : R) {} // expected-warning {{'for co_await' belongs to CoroutineTS instead of C++20, which is deprecated}}
  // expected-error@-1 {{overload resolution selected deleted operator 'co_await'}}
  // expected-note@-2 {{in implicit call to 'operator++' for iterator of type 'Range<long>'}}
}
template ForLoopAwaiterBadIncTransform bad_inc_transform_template(long); // expected-note {{requested here}}

// Ensure we mark and check the function as a coroutine even if it's
// never instantiated.
template <class T>
constexpr void never_instant(T) {
  static_assert(sizeof(T) != sizeof(T), "function should not be instantiated");
  for co_await(auto i : foo(T{})) {} // expected-warning {{'for co_await' belongs to CoroutineTS instead of C++20, which is deprecated}}
  // expected-error@-1 {{'co_await' cannot be used in a constexpr function}}
}

namespace NS {
struct ForLoopAwaiterCoawaitLookup {
  struct promise_type {
    ForLoopAwaiterCoawaitLookup get_return_object();
    void return_void();
    void unhandled_exception();
    suspend_never initial_suspend();
    suspend_never final_suspend() noexcept;
    template <class T>
    CoawaitTag<T, false> await_transform(BeginTag<T> e);
    template <class T>
    Awaiter<T> await_transform(IncTag<T>);
  };
};
} // namespace NS
using NS::ForLoopAwaiterCoawaitLookup;

template <class T>
ForLoopAwaiterCoawaitLookup test_coawait_lookup(T) {
  Range<T> R;
  for co_await(auto i : R) {} // expected-warning {{'for co_await' belongs to CoroutineTS instead of C++20, which is deprecated}}
  // expected-error@-1 {{no member named 'await_ready' in 'CoawaitTag<Iter<int>>'}}
}
template ForLoopAwaiterCoawaitLookup test_coawait_lookup(int); // expected-note {{requested here}}

// FIXME: This test should fail as well since the newly declared operator co_await
// should not be found by lookup.
namespace NS2 {
template <class Iter>
Awaiter<Iter> operator co_await(CoawaitTag<Iter, false>);
}
using NS2::operator co_await;
template ForLoopAwaiterCoawaitLookup test_coawait_lookup(long);

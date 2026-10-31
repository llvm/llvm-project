// RUN: %clang_analyze_cc1 -std=c++23 -analyzer-checker=alpha.webkit.BorrowPlacementChecker -verify %s

#include "mock-canborrow.h"
#include "mock-std-lifetime.h"

void someFunction();

namespace allowed {

void local_variable(Vector<int> &v) {
  Borrow b = v;
  Borrow<Vector<int>> b2(v);
  Borrow<Vector<int>> array[2] = {v, v};
  someFunction();
}

void temporary(Vector<int> &v) {
  for (auto &x : borrow(v).get())
    x = 0;
  const auto &extended = borrow(v);
  (void)extended;
}

void takes_reference(const Borrow<Vector<int>> &);
void takes_value(Borrow<Vector<int>>);
Borrow<Vector<int>> returns_value(Vector<int> &v LIFETIME_BOUND);

void parameters(Vector<int> &v) {
  takes_reference(borrow(v));
  takes_value(borrow(v));
  Borrow b = returns_value(v);
}

void lambda_capture_by_reference(Vector<int> &v) {
  Borrow b = v;
  [&b] { someFunction(); }();
}

struct HoldsPointer {
  Borrow<Vector<int>> *pointer;
  Borrow<Vector<int>> &reference;
};

} // namespace allowed

namespace optional {

void reported_case() {
  Vector<int> v;
  v.append(0xAA);

  std::optional<Borrow<Vector<int>>> o;
  // expected-warning@-1{{Variable 'o' stores a Borrow inside 'std::optional<Borrow<Vector<int>>>'; a Borrow may only be a local variable, parameter, or temporary}}
  o.emplace(v);

  for (auto &x : borrow(v).get()) {
    o.reset();
    x = 0xBB;
  }
}

void parameter(std::optional<Borrow<Vector<int>>> o);
// expected-warning@-1{{Variable 'o' stores a Borrow inside 'std::optional<Borrow<Vector<int>>>'}}

std::optional<Borrow<Vector<int>>> make(Vector<int> &v);
// expected-warning@-1{{Return value of 'make' stores a Borrow inside 'std::optional<Borrow<Vector<int>>>'}}

void temporary(Vector<int> &v) {
  for (auto &x : (*make(v)).get())
    // expected-warning@-1{{Temporary stores a Borrow inside 'std::optional<Borrow<Vector<int>>>'}}
    x = 0;
}

} // namespace optional

namespace members {

struct HasBorrow {
  Borrow<Vector<int>> b;
  // expected-warning@-1{{Member 'b' is a Borrow}}
};

union HasBorrowUnion {
  HasBorrowUnion() {}
  ~HasBorrowUnion() {}
  Borrow<Vector<int>> b;
  // expected-warning@-1{{Member 'b' is a Borrow}}
};

struct HasOptional {
  std::optional<Borrow<Vector<int>>> o;
  // expected-warning@-1{{Member 'o' stores a Borrow inside 'std::optional<Borrow<Vector<int>>>'}}
};

void aggregate(Vector<int> &v) {
  HasBorrow h { v };
  // expected-warning@-1{{Variable 'h' stores a Borrow inside 'HasBorrow'}}
}

template <typename T> struct Holder {
  T value;
  // expected-warning@-1{{Member 'value' is a Borrow}}
};

void template_argument(Vector<int> &v) {
  Holder<Borrow<Vector<int>>> h { v };
  // expected-warning@-1{{Variable 'h' stores a Borrow inside 'Holder<Borrow<Vector<int>>>'}}
}

void unique_ptr(Borrow<Vector<int>> *p) {
  std::unique_ptr<Borrow<Vector<int>>> owner(p);
  // expected-warning@-1{{Variable 'owner' stores a Borrow inside 'std::unique_ptr<Borrow<Vector<int>>>'}}
}

void lambda_capture_by_value(Vector<int> &v) {
  auto lambda = [b = borrow(v)] { someFunction(); };
  // expected-warning@-1{{Lambda capture 'b' is stored in the closure object}}
  // expected-warning@-2{{Variable 'lambda' stores a Borrow inside}}
  lambda();
}

} // namespace members

namespace storage_duration {

Vector<int> globalVector;
Borrow<Vector<int>> globalBorrow(globalVector);
// expected-warning@-1{{Variable 'globalBorrow' has static or thread storage duration}}

void static_local() {
  static Borrow<Vector<int>> b(globalVector);
  // expected-warning@-1{{Variable 'b' has static or thread storage duration}}
}

void thread_local_variable() {
  thread_local Borrow<Vector<int>> b(globalVector);
  // expected-warning@-1{{Variable 'b' has static or thread storage duration}}
}

struct Task {
  struct promise_type {
    Task get_return_object();
    std::suspend_never initial_suspend();
    std::suspend_never final_suspend() noexcept;
    void return_void();
    void unhandled_exception();
  };
};

Task coroutine(Vector<int> &v) {
  Borrow b = v;
  // expected-warning@-1{{Variable 'b' lives in a coroutine frame}}
  co_await std::suspend_never {};
}

} // namespace storage_duration

namespace manual_lifetime {

void placement_new(Vector<int> &v) {
  alignas(Borrow<Vector<int>>) char storage[sizeof(Borrow<Vector<int>>)];
  auto *b = ::new (static_cast<void *>(storage)) Borrow<Vector<int>>(v);
  // expected-warning@-1{{New expression creates a Borrow; a Borrow may only be a local variable, parameter, or temporary}}
  b->~Borrow();
  // expected-warning@-1{{Explicit destructor call ends a Borrow out of order}}
}

void std_lifetime_functions(Vector<int> &v) {
  alignas(Borrow<Vector<int>>) char storage[sizeof(Borrow<Vector<int>>)];
  auto *p = reinterpret_cast<Borrow<Vector<int>> *>(storage);
  std::construct_at(p, v);
  // expected-warning@-1{{Call to 'std::construct_at' manages the lifetime of a Borrow}}
  std::destroy_at(p);
  // expected-warning@-1{{Call to 'std::destroy_at' manages the lifetime of a Borrow}}
}

} // namespace manual_lifetime

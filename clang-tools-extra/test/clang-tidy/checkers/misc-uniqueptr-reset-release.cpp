// RUN: %check_clang_tidy %s misc-uniqueptr-reset-release %t

// CHECK-FIXES: #include <utility>

namespace std {

template <typename T>
struct default_delete {};

template <typename T, class Deleter = std::default_delete<T>>
struct unique_ptr {
  unique_ptr();
  explicit unique_ptr(T *);
  template <typename U, typename E>
  unique_ptr(unique_ptr<U, E> &&);
  void reset(T *);
  T *release();
};
} // namespace std

struct Foo {};
struct Bar : Foo {};

std::unique_ptr<Foo> Create();
std::unique_ptr<Foo> &Look();
std::unique_ptr<Foo> *Get();

using FooFunc = void (*)(Foo *);
using BarFunc = void (*)(Bar *);

void f() {
  std::unique_ptr<Foo> a, b;
  std::unique_ptr<Bar> c;
  std::unique_ptr<Foo> *x = &a;
  std::unique_ptr<Foo> *y = &b;

  a.reset(b.release());
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: prefer 'unique_ptr<>' assignment over 'release' and 'reset' [misc-uniqueptr-reset-release]
  // CHECK-FIXES: a = std::move(b);
  a.reset(c.release());
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: prefer 'unique_ptr<>' assignment
  // CHECK-FIXES: a = std::move(c);
  a.reset(Create().release());
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: prefer 'unique_ptr<>' assignment
  // CHECK-FIXES: a = Create();
  x->reset(y->release());
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: prefer 'unique_ptr<>' assignment
  // CHECK-FIXES: *x = std::move(*y);
  Look().reset(Look().release());
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: prefer 'unique_ptr<>' assignment
  // CHECK-FIXES: Look() = std::move(Look());
  Get()->reset(Get()->release());
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: prefer 'unique_ptr<>' assignment
  // CHECK-FIXES: *Get() = std::move(*Get());

  std::unique_ptr<Bar, FooFunc> func_a, func_b;
  func_a.reset(func_b.release());
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: prefer 'unique_ptr<>' assignment
  // CHECK-FIXES: func_a = std::move(func_b);
}

void negatives() {
  std::unique_ptr<Foo> src;
  struct OtherDeleter {};
  std::unique_ptr<Foo, OtherDeleter> dest;
  dest.reset(src.release());

  std::unique_ptr<Bar, FooFunc> func_a;
  std::unique_ptr<Bar, BarFunc> func_b;
  func_a.reset(func_b.release());
}

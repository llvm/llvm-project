// RUN: %clang_cc1 -std=c++20 %s -verify

template<typename T, typename U>
U f(T t) {
  co_await t;
  co_yield t;

  1 + co_await t;
  1 + co_yield t; // expected-error {{expected expression}}

  auto x = co_await t;
  auto y = co_yield t;

  for co_await (int x : t) {} // expected-warning {{'for co_await' belongs to CoroutineTS instead of C++20, which is deprecated}}
  for co_await (int x = 0; x != 10; ++x) {} // expected-error {{'co_await' modifier can only be applied to range-based for loop}}

  if (t)
    co_return t;
  else
    co_return {t};
}

struct Y {};
struct X { Y operator co_await(); };
struct Z {};
Y operator co_await(Z);

void f(X x, Z z) {
  x.operator co_await();
  operator co_await(z);
}

void operator co_await(); // expected-error {{must have at least one parameter}}
void operator co_await(X, Y, Z); // expected-error {{must be a unary operator}}
void operator co_await(int); // expected-error {{parameter of class or enumeration type}}

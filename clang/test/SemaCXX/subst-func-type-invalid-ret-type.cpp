// RUN: %clang -fsyntax-only -std=c++17 %s -Xclang -verify

// The important part is that we do not crash.

template<typename T> T declval();

template <typename T>
auto Call(T x) -> decltype(declval<T>()(0)) {}

class Status {};

void fun() {
  // The Status() (instead of Status) here used to cause a crash.
  Call([](auto x) -> Status() {}); // expected-error{{function cannot return function type 'Status ()}}
}

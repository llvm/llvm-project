// RUN: %clang_cc1 -fsyntax-only %s -std=c++98 -verify
// expected-no-diagnostics

// This is a test for a hack in Clang that works around an issue with libc++
// 3.1's std::move and std::forward implementation. When emulating these
// functions in C++98 mode, libc++ 3.1 has a "fake rvalue reference" type, and
// std::move will return by value when given an instance of that type.

namespace std {
  struct rv {};

  template<bool B, typename T> struct enable_if;
  template<typename T> struct enable_if<true, T> { typedef T type; };

  template<typename T> typename enable_if<__is_convertible(T, rv), T>::type move(T &);
  template<typename T> typename enable_if<!__is_convertible(T, rv), T&>::type move(T &);

  template<typename U, typename T> typename enable_if<__is_convertible(T, rv), U>::type forward(T &);
  template<typename U, typename T> typename enable_if<!__is_convertible(T, rv), U&>::type forward(T &);
}

struct A {};
void f(A a, std::rv rv) {
  a = std::move(a);
  rv = std::move(rv);

  a = std::forward<A>(a);
  rv = std::forward<std::rv>(rv);

  a = std::forward<A&>(a);
  rv = std::forward<std::rv&>(rv);
}

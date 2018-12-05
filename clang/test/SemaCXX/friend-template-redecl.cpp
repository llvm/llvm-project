// RUN: %clang_cc1 -std=c++17 -verify -emit-llvm-only %s

// expected-no-diagnostics

template <class T> void bar(const T &t) { foo(t); }

template <class>
struct HasFriend {
  template <class T>
  friend void foo(const HasFriend<T> &m) noexcept(false);
};

template <class T>
void foo(const HasFriend<T> &m) noexcept(false) {}

void f() {
  HasFriend<int> x;
  foo(x);
  bar(x);
}

namespace PR39742 {
template<typename>
struct wrapper {
  template<typename>
  friend void friend_function_template() {}
};

wrapper<bool> x;
wrapper<int> y;
}

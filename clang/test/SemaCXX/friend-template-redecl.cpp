// RUN: %clang_cc1 -std=c++17 -verify -emit-llvm-only %s

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

template<typename T> void droid();
struct X {
  template<typename T> friend void ::droid();
  template<int N> friend void ::droid(); // expected-error {{does not match}}
  // FIXME: We should produce a note for the above candidate explaining why
  // it's not the droid we're looking for.
};

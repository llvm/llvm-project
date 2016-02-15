// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify -std=c++11 %s
// expected-no-diagnostics

namespace PR26599 {
template <typename>
struct S;

struct I {};

template <typename T>
void *&non_pointer() {
  void *&r = S<T>()[I{}];
  return r;
}

template <typename T>
void *&pointer() {
  void *&r = S<T>()[nullptr];
  return r;
}
}


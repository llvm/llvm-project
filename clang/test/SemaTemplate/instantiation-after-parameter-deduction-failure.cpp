// RUN: %clang_cc1 -std=c++23 -fsyntax-only -verify %s

namespace ambiguous_overload {
template <class> struct S {
  template <class T> S(T);
};
struct S1 {};
struct S2 {
  operator S1();
};

template <typename T> auto foo(T, S<decltype(0)>); // #declaration
template <typename T> auto foo(T arg, decltype(arg)) {
  foo(arg, S2{});
  // expected-error@-1 {{function 'foo<ambiguous_overload::S1>' with deduced return type cannot be used before it is defined}}
  // expected-note@#declaration {{'foo<ambiguous_overload::S1>' declared here}}
  // expected-note@#instantiation {{in instantiation of function template specialization 'ambiguous_overload::foo<ambiguous_overload::S1>' requested here}}
}

void bar(S1 d) {
  foo(d, S1{}); // #instantiation
}
}

namespace explicit_specialization {
  template <typename T> void foo(T, int);
  template <typename T> void foo(T arg, decltype(arg));
  template <> void foo(int, int) {}
}

namespace address_of {
  template <typename T> void foo(T, int);
  template <typename T> void foo(T arg, decltype(arg));
  void (*p)(int, int) = foo;
}

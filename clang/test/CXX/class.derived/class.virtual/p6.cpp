// RUN:  %clang_cc1 -std=c++2a -verify %s

template<typename T>
class A {
  virtual void f1() requires (sizeof(T) == 0);
  // expected-error@-1{{virtual function cannot have a requires clause}}
  virtual void f2() requires (sizeof(T) == 1);
  // expected-error@-1{{virtual function cannot have a requires clause}}
};

template<typename T>
class B : A<T> {
  virtual void f1() requires (sizeof(T) == 0) override {}
  // expected-error@-1{{virtual function cannot have a requires clause}}
};

template<typename T> struct C : T {void f() requires true; };
// expected-error@-1{{virtual function cannot have a requires clause}}
struct D { virtual void f(); };
template struct C<D>;
// expected-note@-1{{in instantiation of template class 'C<D>' requested here}}
// RUN: %check_clang_tidy "%s" misc-multiple-inheritance "%t"

template<class T> struct X {
  struct B;
  struct A : public B { virtual void foo() {} };
};
template<class T> struct X<T>::B : public A { virtual void foo() {} };

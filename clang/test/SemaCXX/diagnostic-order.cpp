// RUN: not %clang_cc1 -std=c++11 %s -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang_cc1 -std=c++11 %s -fsyntax-only -DWARN 2>&1 | FileCheck %s --check-prefix=CHECK-WARN

#ifndef WARN

// Ensure that the diagnostics we produce for this situation appear in a
// deterministic order. This requires ADL to provide lookup results in a
// deterministic order.
template<typename T, typename> struct Error { typedef typename T::error error; };
struct X { template<typename T> friend typename Error<X, T>::error f(X, T); };
struct Y { template<typename T> friend typename Error<Y, T>::error f(T, Y); };

void g() {
  f(X(), Y());
}

// We don't really care which order these two diagnostics appear (although the
// order below is source order, which seems best). The crucial fact is that
// there is one single order that is stable across multiple runs of clang.
//
// CHECK: no type named 'error' in 'X'
// CHECK: no type named 'error' in 'Y'
// CHECK: no matching function for call to 'f'


struct Oper {
  template<typename T, typename U = typename Error<Oper, T>::error> operator T();

  operator int*();
  operator float*();
  operator X*();
  operator Y*();

  operator int(*[1])();
  operator int(*[2])();
  operator int(*[3])();
  operator int(*[4])();
  operator int(*[5])();
  operator int(*[6])();
  operator int(*[7])();
  operator int(*[8])();
  operator float(*[1])();
  operator float(*[2])();
  operator float(*[3])();
  operator float(*[4])();
  operator float(*[5])();
  operator float(*[6])();
  operator float(*[7])();
  operator float(*[8])();
};
int *p = Oper() + 0;

// CHECK: no type named 'error' in 'Oper'
// CHECK: in instantiation of template class 'Error<Oper, int *>'
// CHECK: no type named 'error' in 'Oper'
// CHECK: in instantiation of template class 'Error<Oper, float *>'
// CHECK: no type named 'error' in 'Oper'
// CHECK: in instantiation of template class 'Error<Oper, X *>'
// CHECK: no type named 'error' in 'Oper'
// CHECK: in instantiation of template class 'Error<Oper, Y *>'

#endif

template<typename T> struct UndefButUsed {
  static inline int f();
  static int g() { return f(); }
};
int undef_but_used = UndefButUsed<int>::g() + UndefButUsed<float>::g() + UndefButUsed<char>::g() + UndefButUsed<void>::g();

// CHECK-WARN: inline function 'UndefButUsed<int>::f' is not defined
// CHECK-WARN: inline function 'UndefButUsed<float>::f' is not defined
// CHECK-WARN: inline function 'UndefButUsed<char>::f' is not defined
// CHECK-WARN: inline function 'UndefButUsed<void>::f' is not defined

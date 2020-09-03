// RUN: %clang_cc1 -verify -fopenmp -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp-simd -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

template <typename T>
void bar(T &x, T &y) { x.a += y.a; }

namespace N1
{
  struct A { int a; A() : a(0) {} };
  #pragma omp declare reduction(+: A : bar(omp_out, omp_in))
  #pragma omp declare reduction(-: struct A : bar(omp_out, omp_in))
}
// CHECK: namespace N1 {
// CHECK: #pragma omp declare reduction (+ : N1::A : bar(omp_out, omp_in))
// CHECK: #pragma omp declare reduction (- : struct A : bar(omp_out, omp_in))

#pragma omp declare reduction(+ : int, char : omp_out *= omp_in)
// CHECK: #pragma omp declare reduction (+ : int : omp_out *= omp_in){{$}}
// CHECK-NEXT: #pragma omp declare reduction (+ : char : omp_out *= omp_in)


template <class T>
class SSS {
public:
#pragma omp declare reduction(fun : T : omp_out += omp_in) initializer(omp_priv = omp_orig + 15)
#pragma omp declare reduction(fun1 : T : omp_out=1, omp_out=foo(omp_in)) initializer(omp_priv = omp_orig + 14)
  static T foo(T &);
};

// CHECK: template <class T> class SSS {
// CHECK: #pragma omp declare reduction (fun : T : omp_out += omp_in) initializer(omp_priv = omp_orig + 15)
// CHECK: #pragma omp declare reduction (fun1 : T : omp_out = 1 , omp_out = foo(omp_in)) initializer(omp_priv = omp_orig + 14)
// CHECK: };
// CHECK: template<> class SSS<int> {
// CHECK: #pragma omp declare reduction (fun : int : omp_out += omp_in) initializer(omp_priv = omp_orig + 15)
// CHECK: #pragma omp declare reduction (fun1 : int : omp_out = 1 , omp_out = foo(omp_in)) initializer(omp_priv = omp_orig + 14)
// CHECK: };

SSS<int> d;

void init(SSS<int> &lhs, SSS<int> rhs);

#pragma omp declare reduction(fun : SSS < int > : omp_out = omp_in) initializer(init(omp_priv, omp_orig))
// CHECK: #pragma omp declare reduction (fun : SSS<int> : omp_out = omp_in) initializer(init(omp_priv, omp_orig))

// CHECK: template <typename T> T foo(T a) {
// CHECK: #pragma omp declare reduction (fun : T : omp_out += omp_in) initializer(omp_priv = omp_orig + 15);
// CHECK: {
// CHECK: #pragma omp declare reduction (fun : T : omp_out += omp_in) initializer(omp_priv = omp_orig + 15);
// CHECK: }
// CHECK: return a;
// CHECK: }

// CHECK: template<> int foo<int>(int a) {
// CHECK: #pragma omp declare reduction (fun : int : omp_out += omp_in) initializer(omp_priv = omp_orig + 15);
// CHECK: {
// CHECK: #pragma omp declare reduction (fun : int : omp_out += omp_in) initializer(omp_priv = omp_orig + 15);
// CHECK: }
// CHECK: return a;
// CHECK: }
template <typename T>
T foo(T a) {
#pragma omp declare reduction(fun : T : omp_out += omp_in) initializer(omp_priv = omp_orig + 15)
  {
#pragma omp declare reduction(fun : T : omp_out += omp_in) initializer(omp_priv = omp_orig + 15)
  }
  return a;
}

int main() {
  int i = 0;
  SSS<int> sss;
  // TODO: Add support for scoped reduction identifiers
  //  #pragma omp parallel reduction(SSS<int>::fun : i)
  // TODO-CHECK: #pragma omp parallel reduction(SSS<int>::fun: i)
  {
    i += 1;
  }
  // #pragma omp parallel reduction(::fun:sss)
  // TODO-CHECK: #pragma omp parallel reduction(::fun: sss)
  {
  }
  N1::A a;
  // CHECK: #pragma omp parallel reduction(+: a)
  #pragma omp parallel reduction(+: a)
  {
  }
  return foo(15);
}

#endif

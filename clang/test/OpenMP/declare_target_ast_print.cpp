// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -I %S/Inputs -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -std=c++11 -include-pch %t -fsyntax-only -I %S/Inputs -verify %s -ast-print | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=50 -I %S/Inputs -ast-print %s | FileCheck %s --check-prefix=CHECK --check-prefix=OMP50
// RUN: %clang_cc1 -verify -fopenmp -I %S/Inputs -ast-print %s | FileCheck %s --check-prefix=CHECK --check-prefix=OMP51
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=52 -I %S/Inputs -ast-print %s | FileCheck %s --check-prefix=CHECK --check-prefix=OMP52
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -x c++ -std=c++11 -I %S/Inputs -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -std=c++11 -include-pch %t -fsyntax-only -I %S/Inputs -verify %s -ast-print | FileCheck %s --check-prefix=CHECK --check-prefix=OMP50
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -I %S/Inputs -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -std=c++11 -include-pch %t -fsyntax-only -I %S/Inputs -verify %s -ast-print | FileCheck %s --check-prefix=CHECK --check-prefix=OMP51
// RUN: %clang_cc1 -fopenmp -fopenmp-version=52 -x c++ -std=c++11 -I %S/Inputs -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=52 -std=c++11 -include-pch %t -fsyntax-only -I %S/Inputs -verify %s -ast-print | FileCheck %s --check-prefix=CHECK --check-prefix=OMP52

// RUN: %clang_cc1 -verify -fopenmp-simd -I %S/Inputs -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -I %S/Inputs -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -std=c++11 -include-pch %t -fsyntax-only -I %S/Inputs -verify %s -ast-print | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

#if _OPENMP == 201811
void bar();
#pragma omp declare target to(bar) device_type(any)
// OMP50: #pragma omp declare target{{$}}
// OMP50: void bar();
// OMP50: #pragma omp end declare target{{$}}
void baz();
#pragma omp declare target to(baz) device_type(nohost)
// OMP50: #pragma omp declare target device_type(nohost){{$}}
// OMP50: void baz();
// OMP50: #pragma omp end declare target{{$}}
void bazz();
#pragma omp declare target to(bazz) device_type(host)
// OMP50: #pragma omp declare target device_type(host){{$}}
// OMP50: void bazz();
// OMP50: #pragma omp end declare target{{$}}
#endif // _OPENMP

#if _OPENMP == 202011
extern "C" {
void boo_c() {}
#pragma omp declare target to(boo_c) indirect
// OMP51: #pragma omp declare target indirect
// OMP51: void boo_c() {
// OMP51: }
// OMP51: #pragma omp end declare target
#pragma omp declare target indirect
void yoo(){}
#pragma omp end declare target
// OMP51: #pragma omp declare target indirect
// OMP51: void yoo() {
// OMP51: }
// OMP51: #pragma omp end declare target
}
extern "C++" {
void boo_cpp() {}
#pragma omp declare target to(boo_cpp) indirect
// OMP51: #pragma omp declare target indirect
// OMP51: void boo_cpp() {
// OMP51: }
// OMP51: #pragma omp end declare target

constexpr bool f() {return false;}
#pragma omp begin declare target indirect(f())
void zoo() {}
void xoo();
#pragma omp end declare target
#pragma omp declare target to(zoo) indirect(false)
// OMP51: #pragma omp declare target indirect(f())
// OMP51: #pragma omp declare target indirect(false)
// OMP51: void zoo() {
// OMP51: }
// OMP51: #pragma omp end declare target
// OMP51: #pragma omp declare target indirect(f())
// OMP51: void xoo();
// OMP51: #pragma omp end declare target

}
#endif // _OPENMP

#if _OPENMP == 202111
extern "C" {
void boo_c() {}
#pragma omp declare target enter(boo_c) indirect
// OMP52: #pragma omp declare target indirect
// OMP52: void boo_c() {
// OMP52: }
// OMP52: #pragma omp end declare target
#pragma omp declare target indirect
void yoo(){}
#pragma omp end declare target
// OMP52: #pragma omp declare target indirect
// OMP52: void yoo() {
// OMP52: }
// OMP52: #pragma omp end declare target
}
extern "C++" {
void boo_cpp() {}
#pragma omp declare target enter(boo_cpp) indirect
// OMP52: #pragma omp declare target indirect
// OMP52: void boo_cpp() {
// OMP52: }
// OMP52: #pragma omp end declare target

constexpr bool f() {return false;}
#pragma omp begin declare target indirect(f())
void zoo() {}
void xoo();
#pragma omp end declare target
#pragma omp declare target enter(zoo) indirect(false)
// OMP52: #pragma omp declare target indirect(f())
// OMP52: #pragma omp declare target indirect(false)
// OMP52: void zoo() {
// OMP52: }
// OMP52: #pragma omp end declare target
// OMP52: #pragma omp declare target indirect(f())
// OMP52: void xoo();
// OMP52: #pragma omp end declare target

}
#endif // _OPENMP

int out_decl_target = 0;
#pragma omp declare target (out_decl_target)

// CHECK: #pragma omp declare target{{$}}
// CHECK: int out_decl_target = 0;
// CHECK: #pragma omp end declare target{{$}}
// CHECK: #pragma omp declare target{{$}}
// CHECK: void lambda()
// CHECK: #pragma omp end declare target{{$}}

#pragma omp declare target
void lambda () {
#ifdef __cpp_lambdas
  (void)[&] { ++out_decl_target; };
#else
  #pragma clang __debug captured
  (void)out_decl_target;
#endif
};
#pragma omp end declare target

#pragma omp declare target
// CHECK: #pragma omp declare target{{$}}
void foo() {}
// CHECK-NEXT: void foo()
#pragma omp end declare target
// CHECK: #pragma omp end declare target{{$}}

extern "C" {
#pragma omp declare target
// CHECK: #pragma omp declare target
void foo_c() {}
// CHECK-NEXT: void foo_c()
#pragma omp end declare target
// CHECK: #pragma omp end declare target
}

extern "C++" {
#pragma omp declare target
// CHECK: #pragma omp declare target
void foo_cpp() {}
// CHECK-NEXT: void foo_cpp()
#pragma omp end declare target
// CHECK: #pragma omp end declare target
}

#pragma omp declare target
template <class T>
struct C {
// CHECK: template <class T> struct C {
// CHECK: #pragma omp declare target
// CHECK-NEXT: static T ts;
// CHECK-NEXT: #pragma omp end declare target

// CHECK: template<> struct C<int>
  T t;
// CHECK-NEXT: int t;
  static T ts;
// CHECK-NEXT: #pragma omp declare target
// CHECK-NEXT: static int ts;
// CHECK: #pragma omp end declare target

  C(T t) : t(t) {
  }
// CHECK: #pragma omp declare target
// CHECK-NEXT: C(int t) : t(t) {
// CHECK-NEXT: }
// CHECK: #pragma omp end declare target

  T foo() {
    return t;
  }
// CHECK: #pragma omp declare target
// CHECK-NEXT: int foo() {
// CHECK-NEXT: return this->t;
// CHECK-NEXT: }
// CHECK: #pragma omp end declare target
};

template<class T>
T C<T>::ts = 1;
// CHECK: #pragma omp declare target
// CHECK: T ts = 1;
// CHECK: #pragma omp end declare target

// CHECK: #pragma omp declare target
// CHECK: int test1()
int test1() {
  C<int> c(1);
  return c.foo() + c.ts;
}
#pragma omp end declare target
// CHECK: #pragma omp end declare target

int a1;
void f1() {
}
#pragma omp declare target (a1, f1)
// CHECK: #pragma omp declare target{{$}}
// CHECK: int a1;
// CHECK: #pragma omp end declare target{{$}}
// CHECK: #pragma omp declare target{{$}}
// CHECK: void f1()
// CHECK: #pragma omp end declare target{{$}}

int b1, b2, b3;
void f2() {
}
#if _OPENMP == 202111
#pragma omp declare target enter(b1) enter(b2), enter(b3, f2)
#else
#pragma omp declare target to(b1) to(b2), to(b3, f2)
#endif // _OPENMP == 202111
// CHECK: #pragma omp declare target{{$}}
// CHECK: int b1;
// CHECK: #pragma omp end declare target{{$}}
// CHECK: #pragma omp declare target{{$}}
// CHECK: int b2;
// CHECK: #pragma omp end declare target{{$}}
// CHECK: #pragma omp declare target{{$}}
// CHECK: int b3;
// CHECK: #pragma omp end declare target{{$}}
// CHECK: #pragma omp declare target{{$}}
// CHECK: void f2()
// CHECK: #pragma omp end declare target{{$}}

int c1, c2, c3;
#pragma omp declare target link(c1) link(c2), link(c3)
// CHECK: #pragma omp declare target link{{$}}
// CHECK: int c1;
// CHECK: #pragma omp end declare target{{$}}
// CHECK: #pragma omp declare target link{{$}}
// CHECK: int c2;
// CHECK: #pragma omp end declare target{{$}}
// CHECK: #pragma omp declare target link{{$}}
// CHECK: int c3;
// CHECK: #pragma omp end declare target{{$}}

struct SSSt {
#pragma omp declare target
  static int a;
  int b;
#pragma omp end declare target
};

// CHECK: struct SSSt {
// CHECK: #pragma omp declare target
// CHECK: static int a;
// CHECK: #pragma omp end declare target
// CHECK: int b;

template <class T>
struct SSSTt {
#pragma omp declare target
  static T a;
  int b;
#pragma omp end declare target
};

// CHECK: template <class T> struct SSSTt {
// CHECK: #pragma omp declare target
// CHECK: static T a;
// CHECK: #pragma omp end declare target
// CHECK: int b;

#pragma omp declare target
template <typename T>
T baz() { return T(); }
#pragma omp end declare target

template <>
int baz() { return 1; }

// CHECK: #pragma omp declare target
// CHECK: template <typename T> T baz() {
// CHECK:     return T();
// CHECK: }
// CHECK: #pragma omp end declare target
// CHECK: #pragma omp declare target
// CHECK: template<> float baz<float>() {
// CHECK:     return float();
// CHECK: }
// CHECK: template<> int baz<int>() {
// CHECK:     return 1;
// CHECK: }
// CHECK: #pragma omp end declare target

#pragma omp declare target
  #include "declare_target_include.h"
  void xyz();
#pragma omp end declare target

// CHECK: #pragma omp declare target
// CHECK: void zyx();
// CHECK: #pragma omp end declare target
// CHECK: #pragma omp declare target
// CHECK: void xyz();
// CHECK: #pragma omp end declare target

#pragma omp declare target
  #pragma omp declare target
    void abc();
  #pragma omp end declare target
  void cba();
#pragma omp end declare target

// CHECK: #pragma omp declare target
// CHECK: void abc();
// CHECK: #pragma omp end declare target
// CHECK: #pragma omp declare target
// CHECK: void cba();
// CHECK: #pragma omp end declare target

#pragma omp declare target
int abc1() { return 1; }
#if _OPENMP == 202111
#pragma omp declare target enter(abc1) device_type(nohost)
#else
#pragma omp declare target to(abc1) device_type(nohost)
#endif // _OPENMP == 202111
#pragma omp end declare target

// CHECK-NEXT: #pragma omp declare target
// CHECK-NEXT: #pragma omp declare target device_type(nohost)
// CHECK-NEXT: int abc1() {
// CHECK-NEXT: return 1;
// CHECK-NEXT: }
// CHECK-NEXT: #pragma omp end declare target

#pragma omp declare target
int inner_link;
#pragma omp declare target link(inner_link)
#pragma omp end declare target

// CHECK-NEXT: #pragma omp declare target
// CHECK-NEXT: #pragma omp declare target link
// CHECK-NEXT: int inner_link;
// CHECK-NEXT: #pragma omp end declare target

void foo2() { return ;}
// CHECK: #pragma omp declare target
// CHECK-NEXT: void foo2() {
// CHECK-NEXT: return;
// CHECK-NEXT: }

int x;
// CHECK: #pragma omp declare target link
// CHECK-NEXT: int x;
// CHECK-NEXT: #pragma omp end declare target

int main (int argc, char **argv) {
  foo();
  foo_c();
  foo_cpp();
  test1();
  baz<float>();
  baz<int>();

#if _OPENMP == 202111
#pragma omp declare target enter(foo2)
#else
#pragma omp declare target to (foo2)
#endif

  #pragma omp declare target link(x)
  return (0);
}

// CHECK: #pragma omp declare target
// CHECK-NEXT: int ts = 1;
// CHECK-NEXT: #pragma omp end declare target

// Do not expect anything here since the region is empty.
#pragma omp declare target
#pragma omp end declare target

#endif

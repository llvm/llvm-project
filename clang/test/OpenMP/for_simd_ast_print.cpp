// RUN: %clang_cc1 -verify -fopenmp -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp-simd -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

struct S1 {
  S1(): a(0) {}
  S1(int v) : a(v) {}
  int a;
  typedef int type;
};

template <typename T>
class S7 : public T {
protected:
  T a;
  S7() : a(0) {}

public:
  S7(typename T::type v) : a(v) {
#pragma omp for simd private(a) private(this->a) private(T::a)
    for (int k = 0; k < a.a; ++k)
      ++this->a.a;
  }
  S7 &operator=(S7 &s) {
#pragma omp for simd private(a) private(this->a)
    for (int k = 0; k < s.a.a; ++k)
      ++s.a.a;
    return *this;
  }
};

// CHECK: #pragma omp for simd private(this->a) private(this->a) private(T::a){{$}}
// CHECK: #pragma omp for simd private(this->a) private(this->a)
// CHECK: #pragma omp for simd private(this->a) private(this->a) private(this->S1::a)

class S8 : public S7<S1> {
  S8() {}

public:
  S8(int v) : S7<S1>(v){
#pragma omp for simd private(a) private(this->a) private(S7<S1>::a) 
    for (int k = 0; k < a.a; ++k)
      ++this->a.a;
  }
  S8 &operator=(S8 &s) {
#pragma omp for simd private(a) private(this->a)
    for (int k = 0; k < s.a.a; ++k)
      ++s.a.a;
    return *this;
  }
};

// CHECK: #pragma omp for simd private(this->a) private(this->a) private(this->S7<S1>::a)
// CHECK: #pragma omp for simd private(this->a) private(this->a)

void foo() {}
int g_ind = 1;
template<class T, class N> T reduct(T* arr, N num) {
  N i;
  N ind;
  N myind;
  T sum = (T)0;
// CHECK: T sum = (T)0;
#pragma omp for simd private(myind, g_ind), linear(ind), aligned(arr) ordered
// CHECK-NEXT: #pragma omp for simd private(myind,g_ind) linear(ind) aligned(arr) ordered
  for (i = 0; i < num; ++i) {
    myind = ind;
    T cur = arr[myind];
    ind += g_ind;
    sum += cur;
  }
}

template<class T> struct S {
  S(const T &a)
    :m_a(a)
  {}
  T result(T *v) const {
    T res;
    T val;
    T lin = 0;
// CHECK: T res;
// CHECK: T val;
// CHECK: T lin = 0;
    #pragma omp for simd private(val)  safelen(7) linear(lin : -5) lastprivate(res) simdlen(5)
// CHECK-NEXT: #pragma omp for simd private(val) safelen(7) linear(lin: -5) lastprivate(res) simdlen(5)
    for (T i = 7; i < m_a; ++i) {
      val = v[i-7] + m_a;
      res = val;
      lin -= 5;
    }
    const T clen = 3;
// CHECK: T clen = 3;
    #pragma omp for simd safelen(clen-1) simdlen(clen-1)
// CHECK-NEXT: #pragma omp for simd safelen(clen - 1) simdlen(clen - 1)
    for(T i = clen+2; i < 20; ++i) {
// CHECK-NEXT: for (T i = clen + 2; i < 20; ++i) {
      v[i] = v[v-clen] + 1;
// CHECK-NEXT: v[i] = v[v - clen] + 1;
    }
// CHECK-NEXT: }
    return res;
  }
  ~S()
  {}
  T m_a;
};

template<int LEN> struct S2 {
  static void func(int n, float *a, float *b, float *c) {
    int k1 = 0, k2 = 0;
#pragma omp for simd safelen(LEN) linear(k1,k2:LEN) aligned(a:LEN) simdlen(LEN)
    for(int i = 0; i < n; i++) {
      c[i] = a[i] + b[i];
      c[k1] = a[k1] + b[k1];
      c[k2] = a[k2] + b[k2];
      k1 = k1 + LEN;
      k2 = k2 + LEN;
    }
  }
};

// S2<4>::func is called below in main.
// CHECK: template<> struct S2<4> {
// CHECK-NEXT: static void func(int n, float *a, float *b, float *c)     {
// CHECK-NEXT:   int k1 = 0, k2 = 0;
// CHECK-NEXT: #pragma omp for simd safelen(4) linear(k1,k2: 4) aligned(a: 4) simdlen(4)
// CHECK-NEXT:   for (int i = 0; i < n; i++) {
// CHECK-NEXT:     c[i] = a[i] + b[i];
// CHECK-NEXT:     c[k1] = a[k1] + b[k1];
// CHECK-NEXT:     c[k2] = a[k2] + b[k2];
// CHECK-NEXT:     k1 = k1 + 4;
// CHECK-NEXT:     k2 = k2 + 4;
// CHECK-NEXT:   }
// CHECK-NEXT: }

int main (int argc, char **argv) {
  int b = argc, c, d, e, f, g;
  int k1=0,k2=0;
  static int *a;
// CHECK: static int *a;
#pragma omp for simd ordered
// CHECK-NEXT: #pragma omp for simd ordered
  for (int i=0; i < 2; ++i)*a=2;
// CHECK-NEXT: for (int i = 0; i < 2; ++i)
// CHECK-NEXT: *a = 2;
#pragma omp  parallel
#pragma omp for simd private(argc, b),lastprivate(d,f) collapse(2) aligned(a : 4) ,firstprivate( g )
  for (int i = 0; i < 10; ++i)
  for (int j = 0; j < 10; ++j) {foo(); k1 += 8; k2 += 8;}
// CHECK-NEXT: #pragma omp parallel
// CHECK-NEXT: #pragma omp for simd private(argc,b) lastprivate(d,f) collapse(2) aligned(a: 4) firstprivate(g)
// CHECK-NEXT: for (int i = 0; i < 10; ++i)
// CHECK-NEXT: for (int j = 0; j < 10; ++j) {
// CHECK-NEXT: foo();
// CHECK-NEXT: k1 += 8;
// CHECK-NEXT: k2 += 8;
// CHECK-NEXT: }
  for (int i = 0; i < 10; ++i)foo();
// CHECK-NEXT: for (int i = 0; i < 10; ++i)
// CHECK-NEXT: foo();
  const int CLEN = 4;
// CHECK-NEXT: const int CLEN = 4;
  #pragma omp for simd aligned(a:CLEN) linear(a:CLEN) safelen(CLEN) collapse( 1 ) simdlen(CLEN)
// CHECK-NEXT: #pragma omp for simd aligned(a: CLEN) linear(a: CLEN) safelen(CLEN) collapse(1) simdlen(CLEN)
  for (int i = 0; i < 10; ++i)foo();
// CHECK-NEXT: for (int i = 0; i < 10; ++i)
// CHECK-NEXT: foo();

  float arr[16];
  S2<4>::func(0,arr,arr,arr);
  return (0);
}

#endif

// RUN: %clang_cc1 -verify -fopenmp -ast-print %s | FileCheck %s --check-prefix=CHECK --check-prefix=OMP45
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s --check-prefix=CHECK --check-prefix=OMP45
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=50 -ast-print %s -DOMP5 | FileCheck %s --check-prefix=CHECK --check-prefix=OMP50
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -x c++ -std=c++11 -emit-pch -o %t %s -DOMP5
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print -DOMP5 | FileCheck %s --check-prefix=CHECK --check-prefix=OMP50

// RUN: %clang_cc1 -verify -fopenmp-simd -ast-print %s | FileCheck %s --check-prefix=CHECK --check-prefix=OMP45
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s --check-prefix=CHECK --check-prefix=OMP45
// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=50 -ast-print %s -DOMP5 | FileCheck %s --check-prefix=CHECK --check-prefix=OMP50
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -x c++ -std=c++11 -emit-pch -o %t %s -DOMP5
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print -DOMP5 | FileCheck %s --check-prefix=CHECK --check-prefix=OMP50
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

struct SS {
  SS(): a(0) {}
  SS(int v) : a(v) {}
  int a;
  typedef int type;
};

template <typename T>
class S7 : public T {
protected:
  T *a;
  T b[2];
  S7() : a(0) {}

public:
  S7(typename T::type &v) : a((T*)&v) {
#pragma omp simd aligned(a)
    for (int k = 0; k < a->a; ++k)
      ++this->a->a;
  }
  S7 &operator=(S7 &s) {
#pragma omp simd aligned(this->b : 8)
    for (int k = 0; k < s.a->a; ++k)
      ++s.a->a;
    return *this;
  }
};

// CHECK: #pragma omp simd aligned(this->a){{$}}
// CHECK: #pragma omp simd aligned(this->b: 8)
// CHECK: #pragma omp simd aligned(this->a)

class S8 : public S7<SS> {
  S8() {}

public:
  S8(int v) : S7<SS>(v){
#pragma omp simd aligned(S7<SS>::a)
    for (int k = 0; k < a->a; ++k)
      ++this->a->a;
  }
  S8 &operator=(S8 &s) {
#pragma omp simd aligned(this->b: 4)
    for (int k = 0; k < s.a->a; ++k)
      ++s.a->a;
    return *this;
  }
};

// CHECK: #pragma omp simd aligned(this->S7<SS>::a)
// CHECK: #pragma omp simd aligned(this->b: 4)

void foo() {}
int g_ind = 1;
template<class T, class N> T reduct(T* arr, N num) {
  N i;
  N ind;
  N myind;
  N &ref = i;
  T sum = (T)0;
// CHECK: T sum = (T)0;
#pragma omp simd private(myind, g_ind), linear(ind), aligned(arr), linear(uval(ref))
// CHECK-NEXT: #pragma omp simd private(myind,g_ind) linear(ind) aligned(arr) linear(uval(ref))
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
    T &ref = res;
// CHECK: T res;
// CHECK: T val;
// CHECK: T lin = 0;
// CHECK: T &ref = res;
    #pragma omp simd allocate(res) private(val)  safelen(7) linear(lin : -5) lastprivate(res) simdlen(5) linear(ref(ref))
// CHECK-NEXT: #pragma omp simd allocate(res) private(val) safelen(7) linear(lin: -5) lastprivate(res) simdlen(5) linear(ref(ref))
    for (T i = 7; i < m_a; ++i) {
      val = v[i-7] + m_a;
      res = val;
      lin -= 5;
    }
    const T clen = 3;
// CHECK: T clen = 3;
    #pragma omp simd safelen(clen-1) simdlen(clen-1)
// CHECK-NEXT: #pragma omp simd safelen(clen - 1) simdlen(clen - 1)
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
#pragma omp simd safelen(LEN) linear(k1,k2:LEN) aligned(a:LEN) simdlen(LEN) allocate(k1)
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
// CHECK-NEXT: #pragma omp simd safelen(4) linear(k1,k2: 4) aligned(a: 4) simdlen(4) allocate(k1)
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
  int &ref = b;
  static int *a;
// CHECK: static int *a;
#pragma omp simd
// CHECK-NEXT: #pragma omp simd{{$}}
  for (int i=0; i < 2; ++i)*a=2;
// CHECK-NEXT: for (int i = 0; i < 2; ++i)
// CHECK-NEXT: *a = 2;
#ifdef OMP5
#pragma omp simd private(argc, b),lastprivate(d,f) collapse(2) aligned(a : 4) if(simd:a) nontemporal(argc, c, d) lastprivate(conditional: e, f) order(concurrent)
// OMP50-NEXT: #pragma omp simd private(argc,b) lastprivate(d,f) collapse(2) aligned(a: 4) if(simd: a) nontemporal(argc,c,d) lastprivate(conditional: e,f) order(concurrent)
#else
#pragma omp simd private(argc, b),lastprivate(d,f) collapse(2) aligned(a : 4)
// OMP45-NEXT: #pragma omp simd private(argc,b) lastprivate(d,f) collapse(2) aligned(a: 4)
#endif // OMP5
  for (int i = 0; i < 10; ++i)
  for (int j = 0; j < 10; ++j) {foo(); k1 += 8; k2 += 8;}
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
#ifdef OMP5
  #pragma omp simd aligned(a:CLEN) linear(a:CLEN) safelen(CLEN) collapse( 1 ) simdlen(CLEN) linear(val(ref): CLEN) if(a)
// OMP50-NEXT: #pragma omp simd aligned(a: CLEN) linear(a: CLEN) safelen(CLEN) collapse(1) simdlen(CLEN) linear(val(ref): CLEN) if(a)
#else
  #pragma omp simd aligned(a:CLEN) linear(a:CLEN) safelen(CLEN) collapse( 1 ) simdlen(CLEN) linear(val(ref): CLEN)
// OMP45-NEXT: #pragma omp simd aligned(a: CLEN) linear(a: CLEN) safelen(CLEN) collapse(1) simdlen(CLEN) linear(val(ref): CLEN)
#endif // OMP5
  for (int i = 0; i < 10; ++i)foo();
// CHECK-NEXT: for (int i = 0; i < 10; ++i)
// CHECK-NEXT: foo();

  float arr[16];
  S2<4>::func(0,arr,arr,arr);
  return (0);
}

#endif

// RUN: %clang_cc1 -std=c++11 -verify=expected,omp45 -fopenmp -fopenmp-version=45 -ferror-limit 200 %s -Wuninitialized
// RUN: %clang_cc1 -std=c++11 -verify=expected,omp51 -fopenmp -ferror-limit 200 %s -Wuninitialized

// RUN: %clang_cc1 -std=c++11 -verify=expected,omp45 -fopenmp-simd -fopenmp-version=45 -ferror-limit 200 %s -Wuninitialized
// RUN: %clang_cc1 -std=c++11 -verify=expected,omp51 -fopenmp-simd -ferror-limit 200 %s -Wuninitialized
struct ST {
  int *a;
};
struct SA {
  const int d = 5;
  const int da[5] = { 0 };
  ST e;
  ST g[10];
  int i;
  int &j = i;
  int *k = &j;
  int *&z = k;
  int aa[10];
  void func(int arg) {
#pragma omp target data map(i) use_device_ptr // expected-error {{expected '(' after 'use_device_ptr'}}
    {}
#pragma omp target data map(i) use_device_ptr( // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error {{expected expression}}
    {}
#pragma omp target data map(i) use_device_ptr() // expected-error {{expected expression}}
    {}
#pragma omp target data map(i) use_device_ptr(alloc) // expected-error {{use of undeclared identifier 'alloc'}}
    {}
#pragma omp target data map(i) use_device_ptr(arg // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error {{expected pointer or reference to pointer in 'use_device_ptr' clause}}
    {}
#pragma omp target data map(i) use_device_ptr(i) // expected-error {{expected pointer or reference to pointer in 'use_device_ptr' clause}}
    {}
#pragma omp target data map(i) use_device_ptr(j) // expected-error {{expected pointer or reference to pointer in 'use_device_ptr' clause}}
    {}
#pragma omp target data map(i) use_device_ptr(k) // OK
    {}
#pragma omp target data map(i) use_device_ptr(z) // OK
    {}
#pragma omp target data map(i) use_device_ptr(aa) // expected-error{{expected pointer or reference to pointer in 'use_device_ptr' clause}}
    {}
#pragma omp target data map(i) use_device_ptr(e) // expected-error{{expected pointer or reference to pointer in 'use_device_ptr' clause}}
    {}
#pragma omp target data map(i) use_device_ptr(g) // expected-error{{expected pointer or reference to pointer in 'use_device_ptr' clause}}
    {}
#pragma omp target data map(i) use_device_ptr(k,i,j) // expected-error2 {{expected pointer or reference to pointer in 'use_device_ptr' clause}}
    {}
#pragma omp target data map(i) use_device_ptr(d) // expected-error{{expected pointer or reference to pointer in 'use_device_ptr' clause}}
    {}
#pragma omp target data map(i) use_device_ptr(da) // expected-error{{expected pointer or reference to pointer in 'use_device_ptr' clause}}
    {}
#pragma omp target data map(i) use_device_addr // expected-error {{expected '(' after 'use_device_addr'}} omp45-error {{unexpected OpenMP clause 'use_device_addr' in directive '#pragma omp target data'}}
    {}
#pragma omp target data map(i) use_device_addr( // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error {{expected expression}} omp45-error {{unexpected OpenMP clause 'use_device_addr' in directive '#pragma omp target data'}}
    {}
#pragma omp target data map(i) use_device_addr() // expected-error {{expected expression}} omp45-error {{unexpected OpenMP clause 'use_device_addr' in directive '#pragma omp target data'}}
    {}
#pragma omp target data map(i) use_device_addr(alloc) // expected-error {{use of undeclared identifier 'alloc'}} omp45-error {{unexpected OpenMP clause 'use_device_addr' in directive '#pragma omp target data'}}
    {}
#pragma omp target data map(i) use_device_addr(arg // expected-error {{expected ')'}} expected-note {{to match this '('}} omp45-error {{unexpected OpenMP clause 'use_device_addr' in directive '#pragma omp target data'}}
    {}
#pragma omp target data map(i) use_device_addr(i) // omp45-error {{unexpected OpenMP clause 'use_device_addr' in directive '#pragma omp target data'}}
    {}
#pragma omp target data map(i) use_device_addr(j) // omp45-error {{unexpected OpenMP clause 'use_device_addr' in directive '#pragma omp target data'}}
    {}
#pragma omp target data map(i) use_device_addr(k) // omp45-error {{unexpected OpenMP clause 'use_device_addr' in directive '#pragma omp target data'}}
    {}
#pragma omp target data map(i) use_device_addr(z) // omp45-error {{unexpected OpenMP clause 'use_device_addr' in directive '#pragma omp target data'}}
    {}
#pragma omp target data map(i) use_device_addr(aa) // omp45-error {{unexpected OpenMP clause 'use_device_addr' in directive '#pragma omp target data'}}
    {}
#pragma omp target data map(i) use_device_addr(e) // omp45-error {{unexpected OpenMP clause 'use_device_addr' in directive '#pragma omp target data'}}
    {}
#pragma omp target data map(i) use_device_addr(g) // omp45-error {{unexpected OpenMP clause 'use_device_addr' in directive '#pragma omp target data'}}
    {}
#pragma omp target data map(i) use_device_addr(k,i,j) // omp45-error {{unexpected OpenMP clause 'use_device_addr' in directive '#pragma omp target data'}}
    {}
#pragma omp target data map(i) use_device_addr(d) // omp45-error {{unexpected OpenMP clause 'use_device_addr' in directive '#pragma omp target data'}}
    {}
#pragma omp target data map(i) use_device_addr(da) // omp45-error {{unexpected OpenMP clause 'use_device_addr' in directive '#pragma omp target data'}}
    {}
  return;
 }
};
struct SB {
  unsigned A;
  unsigned B;
  float Arr[100];
  float *Ptr;
  float *foo() {
    return &Arr[0];
  }
};

struct SC {
  unsigned A : 2;
  unsigned B : 3;
  unsigned C;
  unsigned D;
  float Arr[100];
  SB S;
  SB ArrS[100];
  SB *PtrS;
  SB *&RPtrS;
  float *Ptr;

  SC(SB *&_RPtrS) : RPtrS(_RPtrS) {}
};

union SD {
  unsigned A;
  float B;
};

struct S1;
extern S1 a;
class S2 {
  mutable int a;
public:
  S2():a(0) { }
  S2(S2 &s2):a(s2.a) { }
  static float S2s;
  static const float S2sc;
};
const float S2::S2sc = 0;
const S2 b;
const S2 ba[5];
class S3 {
  int a;
public:
  S3():a(0) { }
  S3(S3 &s3):a(s3.a) { }
};
const S3 c;
const S3 ca[5];
extern const int f;
class S4 {
  int a;
  S4();
  S4(const S4 &s4);
public:
  S4(int v):a(v) { }
};
class S5 {
  int a;
  S5():a(0) {}
  S5(const S5 &s5):a(s5.a) { }
public:
  S5(int v):a(v) { }
};

S3 h;
#pragma omp threadprivate(h)

typedef int from;

template <typename T, int I>
T tmain(T argc) {
  const T d = 5;
  const T da[5] = { 0 };
  S4 e(4);
  S5 g(5);
  T i;
  T &j = i;
  T *k = &j;
  T *&z = k;
  T aa[10];
#pragma omp target data map(i) use_device_ptr // expected-error {{expected '(' after 'use_device_ptr'}}
  {}
#pragma omp target data map(i) use_device_ptr( // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error {{expected expression}}
  {}
#pragma omp target data map(i) use_device_ptr() // expected-error {{expected expression}}
  {}
#pragma omp target data map(i) use_device_ptr(alloc) // expected-error {{use of undeclared identifier 'alloc'}}
  {}
#pragma omp target data map(i) use_device_ptr(argc // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error{{expected pointer or reference to pointer in 'use_device_ptr' clause}}
  {}
#pragma omp target data map(i) use_device_ptr(i) // expected-error {{expected pointer or reference to pointer in 'use_device_ptr' clause}}
  {}
#pragma omp target data map(i) use_device_ptr(j) // expected-error {{expected pointer or reference to pointer in 'use_device_ptr' clause}}
  {}
#pragma omp target data map(i) use_device_ptr(k) // OK
  {}
#pragma omp target data map(i) use_device_ptr(z) // OK
  {}
#pragma omp target data map(i) use_device_ptr(aa) // expected-error{{expected pointer or reference to pointer in 'use_device_ptr' clause}}
  {}
#pragma omp target data map(i) use_device_ptr(e) // expected-error{{expected pointer or reference to pointer in 'use_device_ptr' clause}}
  {}
#pragma omp target data map(i) use_device_ptr(g) // expected-error{{expected pointer or reference to pointer in 'use_device_ptr' clause}}
  {}
#pragma omp target data map(i) use_device_ptr(k,i,j) // expected-error2 {{expected pointer or reference to pointer in 'use_device_ptr' clause}}
  {}
#pragma omp target data map(i) use_device_ptr(d) // expected-error{{expected pointer or reference to pointer in 'use_device_ptr' clause}}
  {}
#pragma omp target data map(i) use_device_ptr(da) // expected-error{{expected pointer or reference to pointer in 'use_device_ptr' clause}}
  {}
#pragma omp target data map(i) use_device_addr // expected-error {{expected '(' after 'use_device_addr'}} omp45-error {{unexpected OpenMP clause 'use_device_addr' in directive '#pragma omp target data'}}
  {}
#pragma omp target data map(i) use_device_addr( // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error {{expected expression}} omp45-error {{unexpected OpenMP clause 'use_device_addr' in directive '#pragma omp target data'}}
  {}
#pragma omp target data map(i) use_device_addr() // expected-error {{expected expression}} omp45-error {{unexpected OpenMP clause 'use_device_addr' in directive '#pragma omp target data'}}
  {}
#pragma omp target data map(i) use_device_addr(alloc) // expected-error {{use of undeclared identifier 'alloc'}} omp45-error {{unexpected OpenMP clause 'use_device_addr' in directive '#pragma omp target data'}}
  {}
#pragma omp target data map(i) use_device_addr(argc // expected-error {{expected ')'}} expected-note {{to match this '('}} omp45-error {{unexpected OpenMP clause 'use_device_addr' in directive '#pragma omp target data'}}
  {}
#pragma omp target data map(i) use_device_addr(i) // omp45-error {{unexpected OpenMP clause 'use_device_addr' in directive '#pragma omp target data'}}
  {}
#pragma omp target data map(i) use_device_addr(j) // omp45-error {{unexpected OpenMP clause 'use_device_addr' in directive '#pragma omp target data'}}
  {}
#pragma omp target data map(i) use_device_addr(k) // omp45-error {{unexpected OpenMP clause 'use_device_addr' in directive '#pragma omp target data'}}
  {}
#pragma omp target data map(i) use_device_addr(z) // omp45-error {{unexpected OpenMP clause 'use_device_addr' in directive '#pragma omp target data'}}
  {}
#pragma omp target data map(i) use_device_addr(aa) // omp45-error {{unexpected OpenMP clause 'use_device_addr' in directive '#pragma omp target data'}}
  {}
#pragma omp target data map(i) use_device_addr(e) // omp45-error {{unexpected OpenMP clause 'use_device_addr' in directive '#pragma omp target data'}}
  {}
#pragma omp target data map(i) use_device_addr(g) // omp45-error {{unexpected OpenMP clause 'use_device_addr' in directive '#pragma omp target data'}}
  {}
#pragma omp target data map(i) use_device_addr(k,i,j) // omp45-error {{unexpected OpenMP clause 'use_device_addr' in directive '#pragma omp target data'}}
  {}
#pragma omp target data map(i) use_device_addr(d) // omp45-error {{unexpected OpenMP clause 'use_device_addr' in directive '#pragma omp target data'}}
  {}
#pragma omp target data map(i) use_device_addr(da) // omp45-error {{unexpected OpenMP clause 'use_device_addr' in directive '#pragma omp target data'}}
  {}
  return 0;
}

int main(int argc, char **argv) {
  const int d = 5;
  const int da[5] = { 0 };
  S4 e(4);
  S5 g(5);
  int i;
  int &j = i;
  int *k = &j;
  int *&z = k;
  int aa[10];
#pragma omp target data map(i) use_device_ptr // expected-error {{expected '(' after 'use_device_ptr'}}
  {}
#pragma omp target data map(i) use_device_ptr( // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error {{expected expression}}
  {}
#pragma omp target data map(i) use_device_ptr() // expected-error {{expected expression}}
  {}
#pragma omp target data map(i) use_device_ptr(alloc) // expected-error {{use of undeclared identifier 'alloc'}}
  {}
#pragma omp target data map(i) use_device_ptr(argc // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error {{expected pointer or reference to pointer in 'use_device_ptr' clause}}
  {}
#pragma omp target data map(i) use_device_ptr(i) // expected-error {{expected pointer or reference to pointer in 'use_device_ptr' clause}}
  {}
#pragma omp target data map(i) use_device_ptr(j) // expected-error {{expected pointer or reference to pointer in 'use_device_ptr' clause}}
  {}
#pragma omp target data map(i) use_device_ptr(k) // OK
  {}
#pragma omp target data map(i) use_device_ptr(z) // OK
  {}
#pragma omp target data map(i) use_device_ptr(aa) // expected-error{{expected pointer or reference to pointer in 'use_device_ptr' clause}}
  {}
#pragma omp target data map(i) use_device_ptr(e) // expected-error{{expected pointer or reference to pointer in 'use_device_ptr' clause}}
  {}
#pragma omp target data map(i) use_device_ptr(g) // expected-error{{expected pointer or reference to pointer in 'use_device_ptr' clause}}
  {}
#pragma omp target data map(i) use_device_ptr(k,i,j) // expected-error2 {{expected pointer or reference to pointer in 'use_device_ptr' clause}}
  {}
#pragma omp target data map(i) use_device_ptr(d) // expected-error{{expected pointer or reference to pointer in 'use_device_ptr' clause}}
  {}
#pragma omp target data map(i) use_device_ptr(da) // expected-error{{expected pointer or reference to pointer in 'use_device_ptr' clause}}
  {}
#pragma omp target data map(i) use_device_addr // expected-error {{expected '(' after 'use_device_addr'}} omp45-error {{unexpected OpenMP clause 'use_device_addr' in directive '#pragma omp target data'}}
  {}
#pragma omp target data map(i) use_device_addr( // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error {{expected expression}} omp45-error {{unexpected OpenMP clause 'use_device_addr' in directive '#pragma omp target data'}}
  {}
#pragma omp target data map(i) use_device_addr() // expected-error {{expected expression}} omp45-error {{unexpected OpenMP clause 'use_device_addr' in directive '#pragma omp target data'}}
  {}
#pragma omp target data map(i) use_device_addr(alloc) // expected-error {{use of undeclared identifier 'alloc'}} omp45-error {{unexpected OpenMP clause 'use_device_addr' in directive '#pragma omp target data'}}
  {}
#pragma omp target data map(i) use_device_addr(argc // expected-error {{expected ')'}} expected-note {{to match this '('}} omp45-error {{unexpected OpenMP clause 'use_device_addr' in directive '#pragma omp target data'}}
  {}
#pragma omp target data map(i) use_device_addr(i) // omp45-error {{unexpected OpenMP clause 'use_device_addr' in directive '#pragma omp target data'}}
  {}
#pragma omp target data map(i) use_device_addr(j) // omp45-error {{unexpected OpenMP clause 'use_device_addr' in directive '#pragma omp target data'}}
  {}
#pragma omp target data map(i) use_device_addr(k) // omp45-error {{unexpected OpenMP clause 'use_device_addr' in directive '#pragma omp target data'}}
  {}
#pragma omp target data map(i) use_device_addr(z) // omp45-error {{unexpected OpenMP clause 'use_device_addr' in directive '#pragma omp target data'}}
  {}
#pragma omp target data map(i) use_device_addr(aa) // omp45-error {{unexpected OpenMP clause 'use_device_addr' in directive '#pragma omp target data'}}
  {}
#pragma omp target data map(i) use_device_addr(e) // omp45-error {{unexpected OpenMP clause 'use_device_addr' in directive '#pragma omp target data'}}
  {}
#pragma omp target data map(i) use_device_addr(g) // omp45-error {{unexpected OpenMP clause 'use_device_addr' in directive '#pragma omp target data'}}
  {}
#pragma omp target data map(i) use_device_addr(k,i,j) // omp45-error {{unexpected OpenMP clause 'use_device_addr' in directive '#pragma omp target data'}}
  {}
#pragma omp target data map(i) use_device_addr(d) // omp45-error {{unexpected OpenMP clause 'use_device_addr' in directive '#pragma omp target data'}}
  {}
#pragma omp target data map(i) use_device_addr(da) // omp45-error {{unexpected OpenMP clause 'use_device_addr' in directive '#pragma omp target data'}}
  {}
  return tmain<int, 3>(argc); // expected-note {{in instantiation of function template specialization 'tmain<int, 3>' requested here}}
}

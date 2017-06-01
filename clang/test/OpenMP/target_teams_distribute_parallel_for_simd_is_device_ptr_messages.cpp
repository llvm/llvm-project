// RUN: %clang_cc1 -std=c++11 -verify -fopenmp %s
struct ST {
  int *a;
};
typedef int arr[10];
typedef ST STarr[10];
struct SA {
  const int d = 5;
  const int da[5] = { 0 };
  ST e;
  ST g[10];
  STarr &rg = g;
  int i;
  int &j = i;
  int *k = &j;
  int *&z = k;
  int aa[10];
  arr &raa = aa;
  void func(int arg) {
#pragma omp target teams distribute parallel for simd is_device_ptr // expected-error {{expected '(' after 'is_device_ptr'}}
  for (int i=0; i<100; i++)
    ;
#pragma omp target teams distribute parallel for simd is_device_ptr( // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error {{expected expression}}
  for (int i=0; i<100; i++)
    ;
#pragma omp target teams distribute parallel for simd is_device_ptr() // expected-error {{expected expression}}
  for (int i=0; i<100; i++)
    ;
#pragma omp target teams distribute parallel for simd is_device_ptr(alloc) // expected-error {{use of undeclared identifier 'alloc'}}
  for (int i=0; i<100; i++)
    ;
#pragma omp target teams distribute parallel for simd is_device_ptr(arg // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error {{expected pointer, array, reference to pointer, or reference to array in 'is_device_ptr clause'}}
  for (int i=0; i<100; i++)
    ;
#pragma omp target teams distribute parallel for simd is_device_ptr(i) // expected-error {{expected pointer, array, reference to pointer, or reference to array in 'is_device_ptr clause'}}
  for (int i=0; i<100; i++)
    ;
#pragma omp target teams distribute parallel for simd is_device_ptr(j) // expected-error {{expected pointer, array, reference to pointer, or reference to array in 'is_device_ptr clause'}}
  for (int i=0; i<100; i++)
    ;
#pragma omp target teams distribute parallel for simd is_device_ptr(k) // OK
  for (int i=0; i<100; i++)
    ;
#pragma omp target teams distribute parallel for simd is_device_ptr(z) // OK
  for (int i=0; i<100; i++)
    ;
#pragma omp target teams distribute parallel for simd is_device_ptr(aa) // OK
  for (int i=0; i<100; i++)
    ;
#pragma omp target teams distribute parallel for simd is_device_ptr(raa) // OK
  for (int i=0; i<100; i++)
    ;
#pragma omp target teams distribute parallel for simd is_device_ptr(e) // expected-error{{expected pointer, array, reference to pointer, or reference to array in 'is_device_ptr clause'}}
  for (int i=0; i<100; i++)
    ;
#pragma omp target teams distribute parallel for simd is_device_ptr(g) // OK
  for (int i=0; i<100; i++)
    ;
#pragma omp target teams distribute parallel for simd is_device_ptr(rg) // OK
  for (int i=0; i<100; i++)
    ;
#pragma omp target teams distribute parallel for simd is_device_ptr(k,i,j) // expected-error2 {{expected pointer, array, reference to pointer, or reference to array in 'is_device_ptr clause'}}
  for (int i=0; i<100; i++)
    ;
#pragma omp target teams distribute parallel for simd is_device_ptr(d) // expected-error{{expected pointer, array, reference to pointer, or reference to array in 'is_device_ptr clause'}}
  for (int i=0; i<100; i++)
    ;
#pragma omp target teams distribute parallel for simd is_device_ptr(da) // OK
  for (int i=0; i<100; i++)
    ;
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

typedef struct {
  int a;
} S6;

template <typename T, int I>
T tmain(T argc) {
  const T d = 5;
  const T da[5] = { 0 };
  S4 e(4);
  S5 g(5);
  S6 h[10];
  auto &rh = h;
  T i;
  T &j = i;
  T *k = &j;
  T *&z = k;
  T aa[10];
  auto &raa = aa;
  S6 *ps;
#pragma omp target teams distribute parallel for simd is_device_ptr // expected-error {{expected '(' after 'is_device_ptr'}}
  for (int i=0; i<100; i++)
    ;
#pragma omp target teams distribute parallel for simd is_device_ptr( // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error {{expected expression}}
  for (int i=0; i<100; i++)
    ;
#pragma omp target teams distribute parallel for simd is_device_ptr() // expected-error {{expected expression}}
  for (int i=0; i<100; i++)
    ;
#pragma omp target teams distribute parallel for simd is_device_ptr(alloc) // expected-error {{use of undeclared identifier 'alloc'}}
  for (int i=0; i<100; i++)
    ;
#pragma omp target teams distribute parallel for simd is_device_ptr(argc // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error{{expected pointer, array, reference to pointer, or reference to array in 'is_device_ptr clause'}}
  for (int i=0; i<100; i++)
    ;
#pragma omp target teams distribute parallel for simd is_device_ptr(i) // expected-error {{expected pointer, array, reference to pointer, or reference to array in 'is_device_ptr clause'}}
  for (int i=0; i<100; i++)
    ;
#pragma omp target teams distribute parallel for simd is_device_ptr(j) // expected-error {{expected pointer, array, reference to pointer, or reference to array in 'is_device_ptr clause'}}
  for (int i=0; i<100; i++)
    ;
#pragma omp target teams distribute parallel for simd is_device_ptr(k) // OK
  for (int i=0; i<100; i++)
    ;
#pragma omp target teams distribute parallel for simd is_device_ptr(z) // OK
  for (int i=0; i<100; i++)
    ;
#pragma omp target teams distribute parallel for simd is_device_ptr(aa) // OK
  for (int i=0; i<100; i++)
    ;
#pragma omp target teams distribute parallel for simd is_device_ptr(raa) // OK
  for (int i=0; i<100; i++)
    ;
#pragma omp target teams distribute parallel for simd is_device_ptr(e) // expected-error{{expected pointer, array, reference to pointer, or reference to array in 'is_device_ptr clause'}}
  for (int i=0; i<100; i++)
    ;
#pragma omp target teams distribute parallel for simd is_device_ptr(g) // expected-error{{expected pointer, array, reference to pointer, or reference to array in 'is_device_ptr clause'}}
  for (int i=0; i<100; i++)
    ;
#pragma omp target teams distribute parallel for simd is_device_ptr(h) // OK
  for (int i=0; i<100; i++)
    ;
#pragma omp target teams distribute parallel for simd is_device_ptr(rh) // OK
  for (int i=0; i<100; i++)
    ;
#pragma omp target teams distribute parallel for simd is_device_ptr(k,i,j) // expected-error2 {{expected pointer, array, reference to pointer, or reference to array in 'is_device_ptr clause'}}
  for (int i=0; i<100; i++)
    ;
#pragma omp target teams distribute parallel for simd is_device_ptr(d) // expected-error{{expected pointer, array, reference to pointer, or reference to array in 'is_device_ptr clause'}}
  for (int i=0; i<100; i++)
    ;
#pragma omp target teams distribute parallel for simd is_device_ptr(da) // OK
  for (int i=0; i<100; i++)
    ;
#pragma omp target teams distribute parallel for simd map(ps) is_device_ptr(ps) // expected-error{{variable already marked as mapped in current construct}} expected-note{{used here}}
  for (int i=0; i<100; i++)
    ;
#pragma omp target teams distribute parallel for simd is_device_ptr(ps) map(ps) // expected-error{{variable already marked as mapped in current construct}} expected-note{{used here}}
  for (int i=0; i<100; i++)
    ;
#pragma omp target teams distribute parallel for simd map(ps->a) is_device_ptr(ps) // expected-error{{variable already marked as mapped in current construct}} expected-note{{used here}}
  for (int i=0; i<100; i++)
    ;
#pragma omp target teams distribute parallel for simd is_device_ptr(ps) map(ps->a) // expected-error{{pointer cannot be mapped along with a section derived from itself}} expected-note{{used here}}
  for (int i=0; i<100; i++)
    ;
#pragma omp target teams distribute parallel for simd is_device_ptr(ps) firstprivate(ps) // expected-error{{firstprivate variable cannot be in a is_device_ptr clause in '#pragma omp target teams distribute parallel for simd' directive}}
  for (int i=0; i<100; i++)
    ;
#pragma omp target teams distribute parallel for simd firstprivate(ps) is_device_ptr(ps) // expected-error{{firstprivate variable cannot be in a is_device_ptr clause in '#pragma omp target teams distribute parallel for simd' directive}} expected-note{{defined as firstprivate}}
  for (int i=0; i<100; i++)
    ;
#pragma omp target teams distribute parallel for simd is_device_ptr(ps) private(ps) // expected-error{{private variable cannot be in a is_device_ptr clause in '#pragma omp target teams distribute parallel for simd' directive}}
  for (int i=0; i<100; i++)
    ;
#pragma omp target teams distribute parallel for simd private(ps) is_device_ptr(ps) // expected-error{{private variable cannot be in a is_device_ptr clause in '#pragma omp target teams distribute parallel for simd' directive}} expected-note{{defined as private}}
  for (int i=0; i<100; i++)
    ;
  return 0;
}

int main(int argc, char **argv) {
  const int d = 5;
  const int da[5] = { 0 };
  S4 e(4);
  S5 g(5);
  S6 h[10];
  auto &rh = h;
  int i;
  int &j = i;
  int *k = &j;
  int *&z = k;
  int aa[10];
  auto &raa = aa;
  S6 *ps;
#pragma omp target teams distribute parallel for simd is_device_ptr // expected-error {{expected '(' after 'is_device_ptr'}}
  for (int i=0; i<100; i++)
    ;
#pragma omp target teams distribute parallel for simd is_device_ptr( // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error {{expected expression}}
  for (int i=0; i<100; i++)
    ;
#pragma omp target teams distribute parallel for simd is_device_ptr() // expected-error {{expected expression}}
  for (int i=0; i<100; i++)
    ;
#pragma omp target teams distribute parallel for simd is_device_ptr(alloc) // expected-error {{use of undeclared identifier 'alloc'}}
  for (int i=0; i<100; i++)
    ;
#pragma omp target teams distribute parallel for simd is_device_ptr(argc // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error {{expected pointer, array, reference to pointer, or reference to array in 'is_device_ptr clause'}}
  for (int i=0; i<100; i++)
    ;
#pragma omp target teams distribute parallel for simd is_device_ptr(i) // expected-error {{expected pointer, array, reference to pointer, or reference to array in 'is_device_ptr clause'}}
  for (int i=0; i<100; i++)
    ;
#pragma omp target teams distribute parallel for simd is_device_ptr(j) // expected-error {{expected pointer, array, reference to pointer, or reference to array in 'is_device_ptr clause'}}
  for (int i=0; i<100; i++)
    ;
#pragma omp target teams distribute parallel for simd is_device_ptr(k) // OK
  for (int i=0; i<100; i++)
    ;
#pragma omp target teams distribute parallel for simd is_device_ptr(z) // OK
  for (int i=0; i<100; i++)
    ;
#pragma omp target teams distribute parallel for simd is_device_ptr(aa) // OK
  for (int i=0; i<100; i++)
    ;
#pragma omp target teams distribute parallel for simd is_device_ptr(raa) // OK
  for (int i=0; i<100; i++)
    ;
#pragma omp target teams distribute parallel for simd is_device_ptr(e) // expected-error{{expected pointer, array, reference to pointer, or reference to array in 'is_device_ptr clause'}}
  for (int i=0; i<100; i++)
    ;
#pragma omp target teams distribute parallel for simd is_device_ptr(g) // expected-error{{expected pointer, array, reference to pointer, or reference to array in 'is_device_ptr clause'}}
  for (int i=0; i<100; i++)
    ;
#pragma omp target teams distribute parallel for simd is_device_ptr(h) // OK
  for (int i=0; i<100; i++)
    ;
#pragma omp target teams distribute parallel for simd is_device_ptr(rh) // OK
  for (int i=0; i<100; i++)
    ;
#pragma omp target teams distribute parallel for simd is_device_ptr(k,i,j) // expected-error2 {{expected pointer, array, reference to pointer, or reference to array in 'is_device_ptr clause'}}
  for (int i=0; i<100; i++)
    ;
#pragma omp target teams distribute parallel for simd is_device_ptr(d) // expected-error{{expected pointer, array, reference to pointer, or reference to array in 'is_device_ptr clause'}}
  for (int i=0; i<100; i++)
    ;
#pragma omp target teams distribute parallel for simd is_device_ptr(da) // OK
  for (int i=0; i<100; i++)
    ;
#pragma omp target teams distribute parallel for simd map(ps) is_device_ptr(ps) // expected-error{{variable already marked as mapped in current construct}} expected-note{{used here}}
  for (int i=0; i<100; i++)
    ;
#pragma omp target teams distribute parallel for simd is_device_ptr(ps) map(ps) // expected-error{{variable already marked as mapped in current construct}} expected-note{{used here}}
  for (int i=0; i<100; i++)
    ;
#pragma omp target teams distribute parallel for simd map(ps->a) is_device_ptr(ps) // expected-error{{variable already marked as mapped in current construct}} expected-note{{used here}}
  for (int i=0; i<100; i++)
    ;
#pragma omp target teams distribute parallel for simd is_device_ptr(ps) map(ps->a) // expected-error{{pointer cannot be mapped along with a section derived from itself}} expected-note{{used here}}
  for (int i=0; i<100; i++)
    ;
#pragma omp target teams distribute parallel for simd is_device_ptr(ps) firstprivate(ps) // expected-error{{firstprivate variable cannot be in a is_device_ptr clause in '#pragma omp target teams distribute parallel for simd' directive}}
  for (int i=0; i<100; i++)
    ;
#pragma omp target teams distribute parallel for simd firstprivate(ps) is_device_ptr(ps) // expected-error{{firstprivate variable cannot be in a is_device_ptr clause in '#pragma omp target teams distribute parallel for simd' directive}} expected-note{{defined as firstprivate}}
  for (int i=0; i<100; i++)
    ;
#pragma omp target teams distribute parallel for simd is_device_ptr(ps) private(ps) // expected-error{{private variable cannot be in a is_device_ptr clause in '#pragma omp target teams distribute parallel for simd' directive}}
  for (int i=0; i<100; i++)
    ;
#pragma omp target teams distribute parallel for simd private(ps) is_device_ptr(ps) // expected-error{{private variable cannot be in a is_device_ptr clause in '#pragma omp target teams distribute parallel for simd' directive}} expected-note{{defined as private}}
  for (int i=0; i<100; i++)
    ;
  return tmain<int, 3>(argc); // expected-note {{in instantiation of function template specialization 'tmain<int, 3>' requested here}}
}

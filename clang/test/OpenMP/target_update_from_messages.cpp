// RUN: %clang_cc1 -verify=expected,le50 -fopenmp -ferror-limit 100 %s -Wno-openmp-mapping -Wuninitialized
// RUN: %clang_cc1 -verify=expected,le45 -fopenmp -fopenmp-version=40 -ferror-limit 100 %s -Wno-openmp-mapping -Wuninitialized
// RUN: %clang_cc1 -verify=expected,le45 -fopenmp -fopenmp-version=45 -ferror-limit 100 %s -Wno-openmp-mapping -Wuninitialized
// RUN: %clang_cc1 -verify=expected,le50 -fopenmp -fopenmp-version=50 -ferror-limit 100 %s -Wno-openmp-mapping -Wuninitialized

// RUN: %clang_cc1 -verify=expected,le50 -fopenmp-simd -ferror-limit 100 %s -Wno-openmp-mapping -Wuninitialized

void foo() {
}

bool foobool(int argc) {
  return argc;
}

struct S1; // expected-note 2 {{declared here}}
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
struct S6 {
  int ii;
  int aa[30];
  float xx;
  double *pp;
};
struct S7 {
  int i;
  int a[50];
  float x;
  S6 s6[5];
  double *p;
  unsigned bfa : 4;
};
struct S8 {
  int *ptr;
  int a;
  struct S7* S;
  void foo() {
#pragma omp target update from(*this) // le45-error {{expected expression containing only member accesses and/or array sections based on named variables}} le45-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
    {}
#pragma omp target update from(*(this->ptr)) // le45-error {{expected expression containing only member accesses and/or array sections based on named variables}} le45-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
#pragma omp target update from(*(this->S->i+this->S->p)) // le45-error {{expected expression containing only member accesses and/or array sections based on named variables}} le45-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
#pragma omp target update from(*(this->S->i+this->S->s6[0].pp)) // le45-error {{expected expression containing only member accesses and/or array sections based on named variables}} le45-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
#pragma omp target update from(*(a+this->ptr)) // le45-error {{expected expression containing only member accesses and/or array sections based on named variables}} le45-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
#pragma omp target update from(*(*(this->ptr)+a+this->ptr)) // le45-error {{expected expression containing only member accesses and/or array sections based on named variables}} le45-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
#pragma omp target update from(*(this+this)) // expected-error {{invalid operands to binary expression ('S8 *' and 'S8 *')}}
  }
};

S3 h;
#pragma omp threadprivate(h) // expected-note 2 {{defined as threadprivate or thread local}}

typedef int to;

template <typename T, int I> // expected-note {{declared here}}
T tmain(T argc) {
  const T d = 5;
  const T da[5] = { 0 };
  S4 e(4);
  S5 g(5);
  T i, t[20];
  T &j = i;
  T *k = &j;
  T x;
  T y;
  T from;
  const T (&l)[5] = da;
  T *m;
  S7 s7;

#pragma omp target update from // expected-error {{expected '(' after 'from'}} expected-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
#pragma omp target update from( // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error {{expected expression}} expected-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
#pragma omp target update from() // expected-error {{expected expression}} expected-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
#pragma omp target update() // expected-warning {{extra tokens at the end of '#pragma omp target update' are ignored}} expected-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
#pragma omp target update from(alloc) // expected-error {{use of undeclared identifier 'alloc'}} expected-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
#pragma omp target update from(x)
#pragma omp target update from(t[:I])
#pragma omp target update from(T) // expected-error {{'T' does not refer to a value}} expected-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
#pragma omp target update from(I) // le50-error 2 {{expected addressable lvalue in 'from' clause}} le45-error 2 {{expected expression containing only member accesses and/or array sections based on named variables}}
#pragma omp target update from(S2::S2s)
#pragma omp target update from(S2::S2sc)
#pragma omp target update from(from)
#pragma omp target update from(y x) // expected-error {{expected ',' or ')' in 'from' clause}}
#pragma omp target update from(argc > 0 ? x : y) // le50-error 2 {{expected addressable lvalue in 'from' clause}} le45-error 2 {{expected expression containing only member accesses and/or array sections based on named variables}}

#pragma omp target update from(S1) // expected-error {{'S1' does not refer to a value}}} expected-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
#pragma omp target update from(a, b, c, d, f) // expected-error {{incomplete type 'S1' where a complete type is required}}
#pragma omp target update from(ba)
#pragma omp target update from(h) // expected-error {{threadprivate variables are not allowed in 'from' clause}} expected-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
#pragma omp target update from(k), to(k) // expected-error 2 {{variable can appear only once in OpenMP 'target update' construct}} expected-note 2 {{used here}}
#pragma omp target update from(t), from(t[:5]) // expected-error 2 {{variable can appear only once in OpenMP 'target update' construct}} expected-note 2 {{used here}}
#pragma omp target update from(da)
#pragma omp target update from(da[:4])

#pragma omp target update from(x, a[:2]) // expected-error {{subscripted value is not an array or pointer}}
#pragma omp target update from(x, c[:]) // expected-error {{subscripted value is not an array or pointer}}
#pragma omp target update from(x, (m+1)[2]) // le45-error 2 {{expected expression containing only member accesses and/or array sections based on named variables}}
#pragma omp target update from(s7.i, s7.a[:3])
#pragma omp target update from(s7.s6[1].aa[0:5])
#pragma omp target update from(x, s7.s6[:5].aa[6]) // expected-error {{OpenMP array section is not allowed here}}
#pragma omp target update from(x, s7.s6[:5].aa[:6]) // expected-error {{OpenMP array section is not allowed here}}
#pragma omp target update from(s7.p[:10])
#pragma omp target update from(x, s7.bfa) // expected-error {{bit fields cannot be used to specify storage in a 'from' clause}}
#pragma omp target update from(x, s7.p[:]) // expected-error {{section length is unspecified and cannot be inferred because subscripted value is not an array}}
#pragma omp target data map(to: s7.i)
  {
#pragma omp target update from(s7.x)
  }

  return 0;
}

int main(int argc, char **argv) {
  const int d = 5;
  const int da[5] = { 0 };
  S4 e(4);
  S5 g(5);
  int i, t[20];
  int &j = i;
  int *k = &j;
  int x;
  int y;
  int from;
  const int (&l)[5] = da;
  int *m;
  S7 s7;
  int **BB, *offset;
  int ***BBB;

#pragma omp target update from // expected-error {{expected '(' after 'from'}} expected-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
#pragma omp target update from( // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error {{expected expression}} expected-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
#pragma omp target update from() // expected-error {{expected expression}} expected-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
#pragma omp target update() // expected-warning {{extra tokens at the end of '#pragma omp target update' are ignored}} expected-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
#pragma omp target update from(alloc) // expected-error {{use of undeclared identifier 'alloc'}} expected-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
#pragma omp target update from(x)
#pragma omp target update from(t[:i])
#pragma omp target update from(S2::S2s)
#pragma omp target update from(S2::S2sc)
#pragma omp target update from(from)
#pragma omp target update from(y x) // expected-error {{expected ',' or ')' in 'from' clause}}
#pragma omp target update from(argc > 0 ? x : y) // le50-error {{expected addressable lvalue in 'from' clause}} le45-error {{expected expression containing only member accesses and/or array sections based on named variables}} expected-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
#pragma omp target update from(S1) // expected-error {{'S1' does not refer to a value}}} expected-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
#pragma omp target update from(a, b, c, d, f) // expected-error {{incomplete type 'S1' where a complete type is required}}
#pragma omp target update from(ba)
#pragma omp target update from(h) // expected-error {{threadprivate variables are not allowed in 'from' clause}} expected-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
#pragma omp target update from(k), to(k) // expected-error {{variable can appear only once in OpenMP 'target update' construct}} expected-note {{used here}}
#pragma omp target update from(t), from(t[:5]) // expected-error {{variable can appear only once in OpenMP 'target update' construct}} expected-note {{used here}}
#pragma omp target update from(da)
#pragma omp target update from(da[:4])

#pragma omp target update from(x, a[:2]) // expected-error {{subscripted value is not an array or pointer}}
#pragma omp target update from(x, c[:]) // expected-error {{subscripted value is not an array or pointer}}
#pragma omp target update from(x, (m+1)[2]) // // le45-error {{expected expression containing only member accesses and/or array sections based on named variables}}
#pragma omp target update from(s7.i, s7.a[:3])
#pragma omp target update from(s7.s6[1].aa[0:5])
#pragma omp target update from(x, s7.s6[:5].aa[6]) // expected-error {{OpenMP array section is not allowed here}}
#pragma omp target update from(x, s7.s6[:5].aa[:6]) // expected-error {{OpenMP array section is not allowed here}}
#pragma omp target update from(s7.p[:10])
#pragma omp target update from(x, s7.bfa) // expected-error {{bit fields cannot be used to specify storage in a 'from' clause}}
#pragma omp target update from(x, s7.p[:]) // expected-error {{section length is unspecified and cannot be inferred because subscripted value is not an array}}
#pragma omp target update from(**(BB+*offset)) // le45-error {{expected expression containing only member accesses and/or array sections based on named variables}} le45-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
#pragma omp target update from(**(BB+y)) // le45-error {{expected expression containing only member accesses and/or array sections based on named variables}} le45-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
#pragma omp target update from(*(m+*offset)) // le45-error {{expected expression containing only member accesses and/or array sections based on named variables}} le45-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
#pragma omp target update from(**(*offset+BB)) // le45-error {{expected expression containing only member accesses and/or array sections based on named variables}} le45-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
#pragma omp target update from(**(y+BB)) // le45-error {{expected expression containing only member accesses and/or array sections based on named variables}} le45-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
#pragma omp target update from(*(*offset+m)) // le45-error {{expected expression containing only member accesses and/or array sections based on named variables}} le45-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
#pragma omp target update from(**(-(*offset)+BB+*m)) // le45-error {{expected expression containing only member accesses and/or array sections based on named variables}} le45-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
#pragma omp target update from(**(*(*(&offset))+BB-*m)) // le45-error {{expected expression containing only member accesses and/or array sections based on named variables}} le45-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
#pragma omp target update from(*(x+*(y+*(**BB+BBB)+s7.i))) // le45-error {{expected expression containing only member accesses and/or array sections based on named variables}} le45-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
#pragma omp target update from(*(m+(m))) // expected-error {{invalid operands to binary expression ('int *' and 'int *')}}
#pragma omp target update from(*(1+y+y)) // expected-error {{indirection requires pointer operand ('int' invalid)}}
#pragma omp target data map(to: s7.i)
  {
#pragma omp target update from(s7.x)
  }

  return tmain<int, 3>(argc)+tmain<to, 4>(argc); // expected-note {{in instantiation of function template specialization 'tmain<int, 3>' requested here}} expected-note {{in instantiation of function template specialization 'tmain<int, 4>' requested here}}
}


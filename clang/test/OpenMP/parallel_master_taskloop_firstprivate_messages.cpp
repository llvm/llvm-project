// RUN: %clang_cc1 -verify -fopenmp %s -Wuninitialized

// RUN: %clang_cc1 -verify -fopenmp-simd %s -Wuninitialized

typedef void **omp_allocator_handle_t;
extern const omp_allocator_handle_t omp_default_mem_alloc;
extern const omp_allocator_handle_t omp_large_cap_mem_alloc;
extern const omp_allocator_handle_t omp_const_mem_alloc;
extern const omp_allocator_handle_t omp_high_bw_mem_alloc;
extern const omp_allocator_handle_t omp_low_lat_mem_alloc;
extern const omp_allocator_handle_t omp_cgroup_mem_alloc;
extern const omp_allocator_handle_t omp_pteam_mem_alloc;
extern const omp_allocator_handle_t omp_thread_mem_alloc;

void foo() {
}

bool foobool(int argc) {
  return argc;
}

void xxx(int argc) {
  int fp; // expected-note {{initialize the variable 'fp' to silence this warning}}
#pragma omp parallel master taskloop firstprivate(fp) // expected-warning {{variable 'fp' is uninitialized when used here}}
  for (int i = 0; i < 10; ++i)
    ;
}

struct S1; // expected-note 2 {{declared here}} expected-note 2 {{forward declaration of 'S1'}}
extern S1 a;
class S2 {
  mutable int a;

public:
  S2() : a(0) {}
  S2(const S2 &s2) : a(s2.a) {}
  static float S2s;
  static const float S2sc;
};
const float S2::S2sc = 0;
const S2 b;
const S2 ba[5];
class S3 {
  int a;
  S3 &operator=(const S3 &s3);

public:
  S3() : a(0) {} // expected-note 2 {{candidate constructor not viable: requires 0 arguments, but 1 was provided}}
  S3(S3 &s3) : a(s3.a) {} // expected-note 2 {{candidate constructor not viable: 1st argument ('const S3') would lose const qualifier}}
};
const S3 c;
const S3 ca[5];
extern const int f;
class S4 {
  int a;
  S4();
  S4(const S4 &s4); // expected-note 2 {{implicitly declared private here}}

public:
  S4(int v) : a(v) {}
};
class S5 {
  int a;
  S5(const S5 &s5) : a(s5.a) {} // expected-note 4 {{implicitly declared private here}}

public:
  S5() : a(0) {}
  S5(int v) : a(v) {}
};
class S6 {
  int a;
  S6() : a(0) {}

public:
  S6(const S6 &s6) : a(s6.a) {}
  S6(int v) : a(v) {}
};

S3 h;
#pragma omp threadprivate(h) // expected-note 2 {{defined as threadprivate or thread local}}

template <class I, class C>
int foomain(int argc, char **argv) {
  I e(4);
  C g(5);
  int i, z;
  int &j = i;
#pragma omp parallel
#pragma omp parallel master taskloop firstprivate // expected-error {{expected '(' after 'firstprivate'}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp parallel
#pragma omp parallel master taskloop firstprivate( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp parallel
#pragma omp parallel master taskloop firstprivate() // expected-error {{expected expression}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp parallel
#pragma omp parallel master taskloop firstprivate(argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp parallel
#pragma omp parallel master taskloop firstprivate(argc, // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp parallel
#pragma omp parallel master taskloop firstprivate(argc > 0 ? argv[1] : argv[2]) // expected-error {{expected variable name}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp parallel
#pragma omp parallel master taskloop allocate(omp_thread_mem_alloc: argc) firstprivate(argc) // expected-warning {{allocator with the 'thread' trait access has unspecified behavior on 'parallel master taskloop' directive}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp parallel
#pragma omp parallel master taskloop firstprivate(S1) // expected-error {{'S1' does not refer to a value}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp parallel
#pragma omp parallel master taskloop firstprivate(a, b) // expected-error {{firstprivate variable with incomplete type 'S1'}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp parallel
#pragma omp parallel master taskloop firstprivate(argv[1]) // expected-error {{expected variable name}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp parallel
#pragma omp parallel master taskloop firstprivate(z, e, g) // expected-error {{calling a private constructor of class 'S4'}} expected-error {{calling a private constructor of class 'S5'}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp parallel
#pragma omp parallel master taskloop firstprivate(h) // expected-error {{threadprivate or thread local variable cannot be firstprivate}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp parallel
  {
    int v = 0;
    int i;
#pragma omp parallel master taskloop firstprivate(i)
    for (int k = 0; k < argc; ++k) {
      i = k;
      v += i;
    }
  }
#pragma omp parallel shared(i)
#pragma omp parallel private(i)
#pragma omp parallel master taskloop firstprivate(j)
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp parallel
#pragma omp parallel master taskloop firstprivate(i)
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp parallel
#pragma omp parallel master taskloop lastprivate(g) firstprivate(g) // expected-error {{calling a private constructor of class 'S5'}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp parallel private(i)
#pragma omp parallel master taskloop firstprivate(i) // expected-note 2 {{defined as firstprivate}}
  for (i = 0; i < argc; ++i) // expected-error 2 {{loop iteration variable in the associated loop of 'omp parallel master taskloop' directive may not be firstprivate, predetermined as private}}
    foo();
#pragma omp parallel reduction(+ : i)  // expected-note {{defined as reduction}}
#pragma omp parallel master taskloop firstprivate(i) // expected-note {{defined as firstprivate}} expected-error {{argument of a reduction clause of a parallel construct must not appear in a firstprivate clause on a task construct}}
  for (i = 0; i < argc; ++i) // expected-error {{loop iteration variable in the associated loop of 'omp parallel master taskloop' directive may not be firstprivate, predetermined as private}}
    foo();
  return 0;
}

void bar(S4 a[2]) {
#pragma omp parallel
#pragma omp parallel master taskloop firstprivate(a)
  for (int i = 0; i < 2; ++i)
    foo();
}

namespace A {
double x;
#pragma omp threadprivate(x) // expected-note {{defined as threadprivate or thread local}}
}
namespace B {
using A::x;
}

int main(int argc, char **argv) {
  const int d = 5;
  const int da[5] = {0};
  S4 e(4);
  S5 g(5);
  S3 m;
  S6 n(2);
  int i;
  int &j = i;
#pragma omp parallel
#pragma omp parallel master taskloop firstprivate // expected-error {{expected '(' after 'firstprivate'}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp parallel
#pragma omp parallel master taskloop firstprivate( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp parallel
#pragma omp parallel master taskloop firstprivate() // expected-error {{expected expression}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp parallel
#pragma omp parallel master taskloop firstprivate(argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp parallel
#pragma omp parallel master taskloop firstprivate(argc, // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp parallel
#pragma omp parallel master taskloop firstprivate(argc > 0 ? argv[1] : argv[2]) // expected-error {{expected variable name}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp parallel
#pragma omp parallel master taskloop firstprivate(argc) allocate , allocate(, allocate(omp_default , allocate(omp_default_mem_alloc, allocate(omp_default_mem_alloc:, allocate(omp_default_mem_alloc: argc, allocate(omp_default_mem_alloc: argv), allocate(argv) // expected-error {{expected '(' after 'allocate'}} expected-error 2 {{expected expression}} expected-error 2 {{expected ')'}} expected-error {{use of undeclared identifier 'omp_default'}} expected-note 2 {{to match this '('}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp parallel
#pragma omp parallel master taskloop firstprivate(S1) // expected-error {{'S1' does not refer to a value}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp parallel
#pragma omp parallel master taskloop firstprivate(a, b, c, d, f) // expected-error {{firstprivate variable with incomplete type 'S1'}} expected-error {{no matching constructor for initialization of 'S3'}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp parallel
#pragma omp parallel master taskloop firstprivate(argv[1]) // expected-error {{expected variable name}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp parallel
#pragma omp parallel master taskloop firstprivate(2 * 2) // expected-error {{expected variable name}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp parallel
#pragma omp parallel master taskloop firstprivate(ba) // OK
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp parallel
#pragma omp parallel master taskloop firstprivate(ca) // expected-error {{no matching constructor for initialization of 'S3'}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp parallel
#pragma omp parallel master taskloop firstprivate(da) // OK
  for (i = 0; i < argc; ++i)
    foo();
  int xa;
#pragma omp parallel
#pragma omp parallel master taskloop firstprivate(xa) // OK
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp parallel
#pragma omp parallel master taskloop firstprivate(S2::S2s) // OK
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp parallel
#pragma omp parallel master taskloop firstprivate(S2::S2sc) // OK
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp parallel
#pragma omp parallel master taskloop safelen(5) // expected-error {{unexpected OpenMP clause 'safelen' in directive '#pragma omp parallel master taskloop'}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp parallel
#pragma omp parallel master taskloop firstprivate(e, g) // expected-error {{calling a private constructor of class 'S4'}} expected-error {{calling a private constructor of class 'S5'}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp parallel
#pragma omp parallel master taskloop firstprivate(m) // OK
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp parallel
#pragma omp parallel master taskloop firstprivate(h) // expected-error {{threadprivate or thread local variable cannot be firstprivate}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp parallel
#pragma omp parallel master taskloop private(xa), firstprivate(xa) // expected-error {{private variable cannot be firstprivate}} expected-note {{defined as private}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp parallel
#pragma omp parallel master taskloop firstprivate(i) // expected-note {{defined as firstprivate}}
  for (i = 0; i < argc; ++i)    // expected-error {{loop iteration variable in the associated loop of 'omp parallel master taskloop' directive may not be firstprivate, predetermined as private}}
    foo();
#pragma omp parallel shared(xa)
#pragma omp parallel master taskloop firstprivate(xa) // OK: may be firstprivate
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp parallel
#pragma omp parallel master taskloop firstprivate(j)
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp parallel
#pragma omp parallel master taskloop lastprivate(g) firstprivate(g) // expected-error {{calling a private constructor of class 'S5'}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp parallel
#pragma omp parallel master taskloop lastprivate(n) firstprivate(n) // OK
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp parallel
  {
    int v = 0;
    int i;
#pragma omp parallel master taskloop firstprivate(i)
    for (int k = 0; k < argc; ++k) {
      i = k;
      v += i;
    }
  }
#pragma omp parallel private(i)
#pragma omp parallel master taskloop firstprivate(i) // expected-note {{defined as firstprivate}}
  for (i = 0; i < argc; ++i) // expected-error {{loop iteration variable in the associated loop of 'omp parallel master taskloop' directive may not be firstprivate, predetermined as private}}
    foo();
#pragma omp parallel reduction(+ : i) // expected-note {{defined as reduction}}
#pragma omp parallel master taskloop firstprivate(i) //expected-error {{argument of a reduction clause of a parallel construct must not appear in a firstprivate clause on a task construct}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp parallel master taskloop firstprivate(i) //expected-note {{defined as firstprivate}}
  for (i = 0; i < argc; ++i) // expected-error {{loop iteration variable in the associated loop of 'omp parallel master taskloop' directive may not be firstprivate, predetermined as private}}
    foo();
#pragma omp parallel
#pragma omp parallel master taskloop firstprivate(B::x) // expected-error {{threadprivate or thread local variable cannot be firstprivate}}
  for (i = 0; i < argc; ++i)
    foo();
  static int si;
#pragma omp parallel master taskloop firstprivate(si) // OK
  for (i = 0; i < argc; ++i)
    si = i + 1;

  return foomain<S4, S5>(argc, argv); // expected-note {{in instantiation of function template specialization 'foomain<S4, S5>' requested here}}
}


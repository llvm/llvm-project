// RUN: %clang_cc1 -verify -fopenmp %s -Wuninitialized

// RUN: %clang_cc1 -verify -fopenmp-simd %s -Wuninitialized

#pragma omp requires dynamic_allocators
typedef void **omp_allocator_handle_t;
extern const omp_allocator_handle_t omp_null_allocator;
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
#pragma omp target parallel for firstprivate(fp) // expected-warning {{variable 'fp' is uninitialized when used here}}
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
  S3() : a(0) {}
  S3(const S3 &s3) : a(s3.a) {}
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
#pragma omp target parallel for firstprivate // expected-error {{expected '(' after 'firstprivate'}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp target parallel for firstprivate( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp target parallel for firstprivate() // expected-error {{expected expression}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp target parallel for firstprivate(argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp target parallel for firstprivate(argc, // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp target parallel for firstprivate(argc > 0 ? argv[1] : argv[2]) // expected-error {{expected variable name}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp target parallel for firstprivate(argc) allocate , allocate(, allocate(omp_default , allocate(omp_default_mem_alloc, allocate(omp_default_mem_alloc:, allocate(omp_default_mem_alloc: argc, allocate(omp_default_mem_alloc: argv), allocate(argv) // expected-error {{expected '(' after 'allocate'}} expected-error 2 {{expected expression}} expected-error 2 {{expected ')'}} expected-error {{use of undeclared identifier 'omp_default'}} expected-note 2 {{to match this '('}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp target parallel for firstprivate(S1) // expected-error {{'S1' does not refer to a value}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp target parallel for firstprivate(z, a, b) // expected-error {{firstprivate variable with incomplete type 'S1'}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp target parallel for firstprivate(argv[1]) // expected-error {{expected variable name}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp target parallel for firstprivate(e, g) // expected-error {{calling a private constructor of class 'S4'}} expected-error {{calling a private constructor of class 'S5'}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp target parallel for firstprivate(h) // expected-error {{threadprivate or thread local variable cannot be firstprivate}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp parallel
  {
    int v = 0;
    int i;
#pragma omp target parallel for firstprivate(i)
    for (int k = 0; k < argc; ++k) {
      i = k;
      v += i;
    }
  }
#pragma omp parallel shared(i)
#pragma omp parallel private(i)
#pragma omp target parallel for firstprivate(j)
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp target parallel for firstprivate(i)
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp target parallel for lastprivate(g) firstprivate(g) // expected-error {{calling a private constructor of class 'S5'}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp parallel private(i)
#pragma omp target parallel for firstprivate(i) // expected-note 2 {{defined as firstprivate}}
  for (i = 0; i < argc; ++i) // expected-error 2 {{loop iteration variable in the associated loop of 'omp target parallel for' directive may not be firstprivate, predetermined as private}}
    foo();
#pragma omp parallel reduction(+ : i)
#pragma omp target parallel for firstprivate(i) // expected-note 2 {{defined as firstprivate}}
  for (i = 0; i < argc; ++i) // expected-error 2 {{loop iteration variable in the associated loop of 'omp target parallel for' directive may not be firstprivate, predetermined as private}}
    foo();
  return 0;
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
  int i, z;
  int &j = i;
#pragma omp target parallel for firstprivate // expected-error {{expected '(' after 'firstprivate'}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp target parallel for firstprivate( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp target parallel for firstprivate() // expected-error {{expected expression}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp target parallel for firstprivate(argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp target parallel for firstprivate(argc, // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp target parallel for firstprivate(argc > 0 ? argv[1] : argv[2]) // expected-error {{expected variable name}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp target parallel for allocate(omp_thread_mem_alloc: argc) firstprivate(argc) // expected-warning {{allocator with the 'thread' trait access has unspecified behavior on 'target parallel for' directive}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp target parallel for firstprivate(S1) // expected-error {{'S1' does not refer to a value}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp target parallel for firstprivate(z, a, b, c, d, f) // expected-error {{firstprivate variable with incomplete type 'S1'}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp target parallel for firstprivate(argv[1]) // expected-error {{expected variable name}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp target parallel for firstprivate(2 * 2) // expected-error {{expected variable name}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp target parallel for firstprivate(ba) // OK
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp target parallel for firstprivate(ca) // OK
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp target parallel for firstprivate(da) // OK
  for (i = 0; i < argc; ++i)
    foo();
  int xa;
#pragma omp target parallel for firstprivate(xa) // OK
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp target parallel for firstprivate(S2::S2s) // OK
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp target parallel for firstprivate(S2::S2sc) // OK
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp target parallel for safelen(5) // expected-error {{unexpected OpenMP clause 'safelen' in directive '#pragma omp target parallel for'}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp target parallel for firstprivate(e, g) // expected-error {{calling a private constructor of class 'S4'}} expected-error {{calling a private constructor of class 'S5'}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp target parallel for firstprivate(m) // OK
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp target parallel for firstprivate(h, B::x) // expected-error 2 {{threadprivate or thread local variable cannot be firstprivate}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp target parallel for private(xa), firstprivate(xa) // expected-error {{private variable cannot be firstprivate}} expected-note {{defined as private}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp target parallel for firstprivate(i) // expected-note {{defined as firstprivate}}
  for (i = 0; i < argc; ++i)    // expected-error {{loop iteration variable in the associated loop of 'omp target parallel for' directive may not be firstprivate, predetermined as private}}
    foo();
#pragma omp parallel shared(xa)
#pragma omp target parallel for firstprivate(xa) // OK: may be firstprivate
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp target parallel for firstprivate(j)
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp target parallel for lastprivate(g) firstprivate(g) // expected-error {{calling a private constructor of class 'S5'}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp target parallel for lastprivate(n) firstprivate(n) // OK
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp parallel
  {
    int v = 0;
    int i;
#pragma omp target parallel for firstprivate(i)
    for (int k = 0; k < argc; ++k) {
      i = k;
      v += i;
    }
  }
#pragma omp parallel private(i)
#pragma omp target parallel for firstprivate(i) // expected-note {{defined as firstprivate}}
  for (i = 0; i < argc; ++i) // expected-error {{loop iteration variable in the associated loop of 'omp target parallel for' directive may not be firstprivate, predetermined as private}}
    foo();
#pragma omp parallel reduction(+ : i)
#pragma omp target parallel for firstprivate(i) // expected-note {{defined as firstprivate}}
  for (i = 0; i < argc; ++i) // expected-error {{loop iteration variable in the associated loop of 'omp target parallel for' directive may not be firstprivate, predetermined as private}}
    foo();
  static int si;
#pragma omp target parallel for firstprivate(si) // OK
  for (i = 0; i < argc; ++i)
    si = i + 1;

  return foomain<S4, S5>(argc, argv); // expected-note {{in instantiation of function template specialization 'foomain<S4, S5>' requested here}}
}

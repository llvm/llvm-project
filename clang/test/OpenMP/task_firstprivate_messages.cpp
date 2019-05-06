// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 %s

// RUN: %clang_cc1 -verify -fopenmp-simd -ferror-limit 100 %s

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

template <typename T>
struct S {
  T b;
  S(T a, T c) {
#pragma omp task default(none) firstprivate(a, b) // expected-note {{explicit data sharing attribute requested here}}
    a = b = c; // expected-error {{variable 'c' must have explicitly specified data sharing attributes}}
  }
};

S<int> s(3, 4); // expected-note {{in instantiation of member function 'S<int>::S' requested here}}

struct S1; // expected-note {{declared here}} expected-note{{forward declaration of 'S1'}}
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
  S4(const S4 &s4); // expected-note 16 {{implicitly declared private here}}

public:
  S4(int v) : a(v) {}
};
class S5 {
  int a;
  S5() : a(0) {}
  S5(const S5 &s5) : a(s5.a) {} // expected-note 16 {{implicitly declared private here}}

public:
  S5(int v) : a(v) {}
};

S3 h;
#pragma omp threadprivate(h) // expected-note {{defined as threadprivate or thread local}}

void bar(int n, int b[n]) {
#pragma omp task firstprivate(b)
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
  int i;
  int &j = i;
  static int m;
#pragma omp task firstprivate                               // expected-error {{expected '(' after 'firstprivate'}}
#pragma omp task firstprivate(                              // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma omp task firstprivate()                             // expected-error {{expected expression}}
#pragma omp task firstprivate(argc                          // expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma omp task firstprivate(argc,                         // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma omp task firstprivate(argc > 0 ? argv[1] : argv[2]) // expected-error {{expected variable name}}
#pragma omp task firstprivate(argc) allocate , allocate(, allocate(omp_default , allocate(omp_default_mem_alloc, allocate(omp_default_mem_alloc:, allocate(omp_default_mem_alloc: argc, allocate(omp_default_mem_alloc: argv), allocate(argv) // expected-error {{expected '(' after 'allocate'}} expected-error 2 {{expected expression}} expected-error 2 {{expected ')'}} expected-error {{use of undeclared identifier 'omp_default'}} expected-note 2 {{to match this '('}}
#pragma omp task firstprivate(S1)            // expected-error {{'S1' does not refer to a value}}
#pragma omp task firstprivate(a, b, c, d, f) // expected-error {{firstprivate variable with incomplete type 'S1'}}
#pragma omp task firstprivate(argv[1])       // expected-error {{expected variable name}}
#pragma omp task allocate(omp_thread_mem_alloc: ba) firstprivate(ba) // expected-warning {{allocator with the 'thread' trait access has unspecified behavior on 'task' directive}}
#pragma omp task firstprivate(ca)
#pragma omp task firstprivate(da)
#pragma omp task firstprivate(S2::S2s)
#pragma omp task firstprivate(S2::S2sc)
#pragma omp task firstprivate(e, g)          // expected-error 16 {{calling a private constructor of class 'S4'}} expected-error 16 {{calling a private constructor of class 'S5'}}
#pragma omp task firstprivate(h, B::x)       // expected-error 2 {{threadprivate or thread local variable cannot be firstprivate}}
#pragma omp task private(i), firstprivate(i) // expected-error {{private variable cannot be firstprivate}} expected-note{{defined as private}}
  foo();
#pragma omp task shared(i)
#pragma omp task firstprivate(i)
#pragma omp task firstprivate(j)
#pragma omp task firstprivate(m) // OK
  foo();

  return 0;
}

// RUN: %clang_cc1 -verify -fopenmp %s -Wuninitialized

// RUN: %clang_cc1 -verify -fopenmp-simd %s -Wuninitialized

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

struct S1; // expected-note 2 {{declared here}} expected-note 2 {{forward declaration of 'S1'}}
extern S1 a;
class S2 {
  mutable int a;

public:
  S2() : a(0) {}
};
const S2 b;
const S2 ba[5];
class S3 {
  int a;

public:
  S3() : a(0) {}
};
const S3 ca[5];
class S4 {
  int a;
  S4(); // expected-note {{implicitly declared private here}}

public:
  S4(int v) : a(v) {
#pragma omp masked taskloop simd private(a) private(this->a)
    for (int k = 0; k < v; ++k)
      ++this->a;
  }
};
class S5 {
  int a;
  S5() : a(0) {} // expected-note {{implicitly declared private here}}

public:
  S5(int v) : a(v) {}
  S5 &operator=(S5 &s) {
#pragma omp masked taskloop simd private(a) private(this->a) private(s.a) // expected-error {{expected variable name or data member of current class}}
    for (int k = 0; k < s.a; ++k)
      ++s.a;
    return *this;
  }
};

template <typename T>
class S6 {
public:
  T a;

  S6() : a(0) {}
  S6(T v) : a(v) {
#pragma omp masked taskloop simd allocate(omp_thread_mem_alloc: a) private(a) private(this->a) // expected-warning {{allocator with the 'thread' trait access has unspecified behavior on 'masked taskloop simd' directive}}
    for (int k = 0; k < v; ++k)
      ++this->a;
  }
  S6 &operator=(S6 &s) {
#pragma omp masked taskloop simd private(a) private(this->a) private(s.a) // expected-error {{expected variable name or data member of current class}}
    for (int k = 0; k < s.a; ++k)
      ++s.a;
    return *this;
  }
};

template <typename T>
class S7 : public T {
  T a;
  S7() : a(0) {}

public:
  S7(T v) : a(v) {
#pragma omp masked taskloop simd private(a) private(this->a) private(T::a)
    for (int k = 0; k < a.a; ++k)
      ++this->a.a;
  }
  S7 &operator=(S7 &s) {
#pragma omp masked taskloop simd private(a) private(this->a) private(s.a) private(s.T::a) // expected-error 2 {{expected variable name or data member of current class}}
    for (int k = 0; k < s.a.a; ++k)
      ++s.a.a;
    return *this;
  }
};

S3 h;
#pragma omp threadprivate(h) // expected-note 2 {{defined as threadprivate or thread local}}

template <class I, class C>
int foomain(I argc, C **argv) {
  I e(4);
  I g(5);
  int i, z;
  int &j = i;
#pragma omp masked taskloop simd private // expected-error {{expected '(' after 'private'}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp masked taskloop simd private( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp masked taskloop simd private() // expected-error {{expected expression}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp masked taskloop simd private(argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp masked taskloop simd private(argc, // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp masked taskloop simd private(argc > 0 ? argv[1] : argv[2]) // expected-error {{expected variable name}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp masked taskloop simd private(argc) allocate , allocate(, allocate(omp_default , allocate(omp_default_mem_alloc, allocate(omp_default_mem_alloc:, allocate(omp_default_mem_alloc: argc, allocate(omp_default_mem_alloc: argv), allocate(argv) // expected-error {{expected '(' after 'allocate'}} expected-error 2 {{expected expression}} expected-error 2 {{expected ')'}} expected-error {{use of undeclared identifier 'omp_default'}} expected-note 2 {{to match this '('}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp masked taskloop simd private(S1) // expected-error {{'S1' does not refer to a value}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp masked taskloop simd private(a, b) // expected-error {{private variable with incomplete type 'S1'}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp masked taskloop simd private(argv[1]) // expected-error {{expected variable name}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp masked taskloop simd private(e, g, z)
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp masked taskloop simd private(h) // expected-error {{threadprivate or thread local variable cannot be private}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp masked taskloop simd shared(i)
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp parallel
  {
    int v = 0;
    int i;
#pragma omp masked taskloop simd private(i)
    for (int k = 0; k < argc; ++k) {
      i = k;
      v += i;
    }
  }
#pragma omp parallel shared(i)
#pragma omp parallel private(i)
#pragma omp masked taskloop simd private(j)
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp masked taskloop simd private(i)
  for (int k = 0; k < argc; ++k)
    ++k;
  return 0;
}

void bar(S4 a[2]) {
#pragma omp parallel
#pragma omp masked taskloop simd private(a)
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
  S4 e(4);
  S5 g(5);
  S6<float> s6(0.0) , s6_0(1.0); // expected-note {{in instantiation of member function 'S6<float>::S6' requested here}}
  S7<S6<float> > s7(0.0) , s7_0(1.0);
  int i, z;
  int &j = i;
  int tid = 0;
#pragma omp masked taskloop simd private filter(tid) // expected-error {{expected '(' after 'private'}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp masked taskloop simd private( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp masked taskloop simd private() // expected-error {{expected expression}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp masked taskloop simd private(argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp masked taskloop simd private(argc, // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp masked taskloop simd private(argc > 0 ? argv[1] : argv[2]) // expected-error {{expected variable name}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp masked taskloop simd private(argc, z)
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp masked taskloop simd private(S1) // expected-error {{'S1' does not refer to a value}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp masked taskloop simd private(a, b) // expected-error {{private variable with incomplete type 'S1'}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp masked taskloop simd private(argv[1]) // expected-error {{expected variable name}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp masked taskloop simd private(e, g) // expected-error {{calling a private constructor of class 'S4'}} expected-error {{calling a private constructor of class 'S5'}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp masked taskloop simd private(h) // expected-error {{threadprivate or thread local variable cannot be private}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp masked taskloop simd private(B::x) // expected-error {{threadprivate or thread local variable cannot be private}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp masked taskloop simd shared(i)
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp parallel
  {
    int i;
#pragma omp masked taskloop simd private(i)
    for (int k = 0; k < argc; ++k)
      ++k;
  }
#pragma omp parallel shared(i)
#pragma omp parallel private(i)
#pragma omp masked taskloop simd private(j)
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp masked taskloop simd private(i)
  for (int k = 0; k < argc; ++k)
    ++k;
  static int si;
#pragma omp masked taskloop simd private(si) // OK
  for(int k = 0; k < argc; ++k)
    si = k + 1;

  s6 = s6_0; // expected-note {{in instantiation of member function 'S6<float>::operator=' requested here}}
  s7 = s7_0; // expected-note {{in instantiation of member function 'S7<S6<float>>::operator=' requested here}}
  return foomain(argc, argv); // expected-note {{in instantiation of function template specialization 'foomain<int, char>' requested here}}
}


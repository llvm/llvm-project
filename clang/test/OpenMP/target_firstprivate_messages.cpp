// RUN: %clang_cc1 -verify -fopenmp %s -Wuninitialized

// RUN: %clang_cc1 -verify -fopenmp-simd %s -Wuninitialized

void xxx(int argc) {
  int fp, fp1; // expected-note {{initialize the variable 'fp' to silence this warning}} expected-note {{initialize the variable 'fp1' to silence this warning}}
#pragma omp target firstprivate(fp) // expected-warning {{variable 'fp' is uninitialized when used here}}
  for (int i = 0; i < 10; ++i)
    ++fp1; // expected-warning {{variable 'fp1' is uninitialized when used here}}
}

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
  S4();

public:
  S4(int v) : a(v) {
#pragma omp target firstprivate(a) firstprivate(this->a)
    for (int k = 0; k < v; ++k)
      ++this->a;
  }
};
class S5 {
  int a;
  S5() : a(0) {}

public:
  S5(int v) : a(v) {}
  S5 &operator=(S5 &s) {
#pragma omp target firstprivate(a) firstprivate(this->a) firstprivate(s.a) // expected-error {{expected variable name or data member of current class}}
    for (int k = 0; k < s.a; ++k) // expected-warning {{Type 'S5' is not trivially copyable and not guaranteed to be mapped correctly}}
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
#pragma omp target firstprivate(a) firstprivate(this->a)
    for (int k = 0; k < v; ++k)
      ++this->a;
  }
  S6 &operator=(S6 &s) {
#pragma omp target firstprivate(a) firstprivate(this->a) firstprivate(s.a) // expected-error {{expected variable name or data member of current class}}
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
#pragma omp target firstprivate(a) firstprivate(this->a) firstprivate(T::a)
    for (int k = 0; k < a.a; ++k)
      ++this->a.a;
  }
  S7 &operator=(S7 &s) {
#pragma omp target firstprivate(a) firstprivate(this->a) firstprivate(s.a) firstprivate(s.T::a) // expected-error 2 {{expected variable name or data member of current class}}
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
#pragma omp target firstprivate // expected-error {{expected '(' after 'firstprivate'}}
{}
#pragma omp target firstprivate( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
{}
#pragma omp target firstprivate() // expected-error {{expected expression}}
{}
#pragma omp target firstprivate(argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
{}
#pragma omp target firstprivate(argc, // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
{}
#pragma omp target firstprivate(argc > 0 ? argv[1] : argv[2]) // expected-error {{expected variable name}}
{}
#pragma omp target firstprivate(argc) allocate , allocate(, allocate(omp_default , allocate(omp_default_mem_alloc, allocate(omp_default_mem_alloc:, allocate(omp_default_mem_alloc: argc, allocate(omp_default_mem_alloc: argv), allocate(argv) // expected-error {{expected '(' after 'allocate'}} expected-error 2 {{expected expression}} expected-error 2 {{expected ')'}} expected-error {{use of undeclared identifier 'omp_default'}} expected-note 2 {{to match this '('}}
{}
#pragma omp target firstprivate(S1) // expected-error {{'S1' does not refer to a value}}
{}
#pragma omp target firstprivate(a, b) // expected-error {{firstprivate variable with incomplete type 'S1'}}
{}
#pragma omp target firstprivate(argv[1]) // expected-error {{expected variable name}}
{}
#pragma omp target firstprivate(e, g) allocate(omp_thread_mem_alloc: e) // expected-warning {{allocator with the 'thread' trait access has unspecified behavior on 'target' directive}} expected-error {{allocator must be specified in the 'uses_allocators' clause}}
{}
#pragma omp target firstprivate(h) // expected-error {{threadprivate or thread local variable cannot be firstprivate}}
{}
#pragma omp target shared(i) // expected-error {{unexpected OpenMP clause 'shared' in directive '#pragma omp target'}}
#pragma omp parallel
  {
    int v = 0;
    int i;
  }
#pragma omp parallel shared(i, z)
#pragma omp parallel firstprivate(i, z)
#pragma omp target firstprivate(j)
{}
#pragma omp target firstprivate(i)
  {}
  return 0;
}

void bar(S4 a[2]) {
#pragma omp parallel
#pragma omp target firstprivate(a)
  {}
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
  S6<float> s6(0.0) , s6_0(1.0);
  S7<S6<float> > s7(0.0) , s7_0(1.0);
  int i, z;
  int &j = i;
#pragma omp target firstprivate // expected-error {{expected '(' after 'firstprivate'}}
{}
#pragma omp target firstprivate( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
{}
#pragma omp target firstprivate() // expected-error {{expected expression}}
{}
#pragma omp target firstprivate(argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
{}
#pragma omp target firstprivate(argc, // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
{}
#pragma omp target firstprivate(argc > 0 ? argv[1] : argv[2]) // expected-error {{expected variable name}}
{}
#pragma omp target firstprivate(argc, z)
{}
#pragma omp target firstprivate(S1) // expected-error {{'S1' does not refer to a value}}
{}
#pragma omp target firstprivate(a, b) // expected-error {{firstprivate variable with incomplete type 'S1'}}
{}
#pragma omp target firstprivate(argv[1]) // expected-error {{expected variable name}}
{}
#pragma omp target firstprivate(e, g)
{}
#pragma omp target firstprivate(h) // expected-error {{threadprivate or thread local variable cannot be firstprivate}}
{}
#pragma omp target firstprivate(B::x) // expected-error {{threadprivate or thread local variable cannot be firstprivate}}
{}
#pragma omp target shared(i) // expected-error {{unexpected OpenMP clause 'shared' in directive '#pragma omp target'}}
#pragma omp parallel
  {
    int i;
  }
#pragma omp parallel shared(i)
#pragma omp parallel firstprivate(i)
#pragma omp target firstprivate(j)
{}
#pragma omp target firstprivate(i)
  {}
  static int si;
#pragma omp target firstprivate(si) // OK
  {}
#pragma omp target map(i) firstprivate(i) // expected-error {{firstprivate variable cannot be in a map clause in '#pragma omp target' directive}}
  {}
  s6 = s6_0; // expected-note {{in instantiation of member function 'S6<float>::operator=' requested here}}
  s7 = s7_0; // expected-note {{in instantiation of member function 'S7<S6<float>>::operator=' requested here}}
  return foomain(argc, argv); // expected-note {{in instantiation of function template specialization 'foomain<int, char>' requested here}}
}


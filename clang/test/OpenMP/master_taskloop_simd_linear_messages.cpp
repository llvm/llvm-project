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

void xxx(int argc) {
  int i, lin, step; // expected-note {{initialize the variable 'lin' to silence this warning}} expected-note {{initialize the variable 'step' to silence this warning}}
#pragma omp master taskloop simd linear(i, lin : step) // expected-warning {{variable 'lin' is uninitialized when used here}} expected-warning {{variable 'step' is uninitialized when used here}}
  for (i = 0; i < 10; ++i)
    ;
}

namespace X {
  int x;
};

struct B {
  static int ib; // expected-note {{'B::ib' declared here}}
  static int bfoo() { return 8; }
};

int bfoo() { return 4; }

int z;
const int C1 = 1;
const int C2 = 2;
void test_linear_colons()
{
  int B = 0;
  #pragma omp master taskloop simd linear(B:bfoo())
  for (int i = 0; i < 10; ++i) ;
  // expected-error@+1 {{unexpected ':' in nested name specifier; did you mean '::'}}
  #pragma omp master taskloop simd linear(B::ib:B:bfoo())
  for (int i = 0; i < 10; ++i) ;
  // expected-error@+1 {{use of undeclared identifier 'ib'; did you mean 'B::ib'}}
  #pragma omp master taskloop simd linear(B:ib)
  for (int i = 0; i < 10; ++i) ;
  // expected-error@+1 {{unexpected ':' in nested name specifier; did you mean '::'?}}
  #pragma omp master taskloop simd linear(z:B:ib)
  for (int i = 0; i < 10; ++i) ;
  #pragma omp master taskloop simd linear(B:B::bfoo())
  for (int i = 0; i < 10; ++i) ;
  #pragma omp master taskloop simd linear(X::x : ::z)
  for (int i = 0; i < 10; ++i) ;
  #pragma omp master taskloop simd linear(B,::z, X::x)
  for (int i = 0; i < 10; ++i) ;
  #pragma omp master taskloop simd linear(::z)
  for (int i = 0; i < 10; ++i) ;
  // expected-error@+1 {{expected variable name}}
  #pragma omp master taskloop simd linear(B::bfoo())
  for (int i = 0; i < 10; ++i) ;
  #pragma omp master taskloop simd linear(B::ib,B:C1+C2)
  for (int i = 0; i < 10; ++i) ;
}

template<int L, class T, class N> T test_template(T* arr, N num) {
  N i;
  T sum = (T)0;
  T ind2 = - num * L; // expected-note {{'ind2' defined here}}
  // expected-error@+1 {{argument of a linear clause should be of integral or pointer type}}
#pragma omp master taskloop simd linear(ind2:L)
  for (i = 0; i < num; ++i) {
    T cur = arr[(int)ind2];
    ind2 += L;
    sum += cur;
  }
  return T();
}

template<int LEN> int test_warn() {
  int ind2 = 0;
  // expected-warning@+1 {{zero linear step (ind2 should probably be const)}}
  #pragma omp master taskloop simd linear(ind2:LEN)
  for (int i = 0; i < 100; i++) {
    ind2 += LEN;
  }
  return ind2;
}

struct S1; // expected-note 2 {{declared here}} expected-note 2 {{forward declaration of 'S1'}}
extern S1 a;
class S2 {
  mutable int a;
public:
  S2():a(0) { }
};
const S2 b; // expected-note 2 {{'b' defined here}}
const S2 ba[5];
class S3 {
  int a;
public:
  S3():a(0) { }
};
const S3 ca[5];
class S4 {
  int a;
  S4();
public:
  S4(int v):a(v) { }
};
class S5 {
  int a;
  S5():a(0) {}
public:
  S5(int v):a(v) { }
};

S3 h;
#pragma omp threadprivate(h) // expected-note 2 {{defined as threadprivate or thread local}}

template<class I, class C> int foomain(I argc, C **argv) {
  I e(4);
  I g(5);
  int i, z;
  int &j = i;
  #pragma omp master taskloop simd linear // expected-error {{expected '(' after 'linear'}}
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp master taskloop simd linear ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp master taskloop simd linear (val // expected-error {{use of undeclared identifier 'val'}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp master taskloop simd linear (uval( // expected-error {{expected expression}} expected-error 2 {{expected ')'}} expected-note 2 {{to match this '('}}
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp master taskloop simd linear (ref() // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp master taskloop simd linear (foo() // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp master taskloop simd linear () // expected-error {{expected expression}}
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp master taskloop simd linear (argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp master taskloop simd linear (val argc // expected-error {{use of undeclared identifier 'val'}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp master taskloop simd linear (val(argc, // expected-error {{expected expression}} expected-error 2 {{expected ')'}} expected-note 2 {{to match this '('}}
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp master taskloop simd linear (argc > 0 ? argv[1] : argv[2]) // expected-error {{expected variable name}}
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp master taskloop simd linear (argc : 5) allocate , allocate(, allocate(omp_default , allocate(omp_default_mem_alloc, allocate(omp_default_mem_alloc:, allocate(omp_default_mem_alloc: argc, allocate(omp_default_mem_alloc: argv), allocate(argv) // expected-error {{expected '(' after 'allocate'}} expected-error 2 {{expected expression}} expected-error 2 {{expected ')'}} expected-error {{use of undeclared identifier 'omp_default'}} expected-note 2 {{to match this '('}}
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp master taskloop simd linear (S1) // expected-error {{'S1' does not refer to a value}}
  for (int k = 0; k < argc; ++k) ++k;
  // expected-error@+2 {{linear variable with incomplete type 'S1'}}
  // expected-error@+1 {{argument of a linear clause should be of integral or pointer type, not 'S2'}}
  #pragma omp master taskloop simd linear (val(a, b):B::ib)
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp master taskloop simd linear (argv[1]) // expected-error {{expected variable name}}
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp master taskloop simd linear(ref(e, g)) // expected-error 2 {{variable of non-reference type 'int' can be used only with 'val' modifier, but used with 'ref'}}
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp master taskloop simd linear(h, z) // expected-error {{threadprivate or thread local variable cannot be linear}}
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp master taskloop simd linear(uval(i)) // expected-error {{variable of non-reference type 'int' can be used only with 'val' modifier, but used with 'uval'}}
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp parallel
  {
    int v = 0;
    int i;
    #pragma omp master taskloop simd allocate(omp_thread_mem_alloc: v) linear(v:i) // expected-warning {{allocator with the 'thread' trait access has unspecified behavior on 'master taskloop simd' directive}}
    for (int k = 0; k < argc; ++k) { i = k; v += i; }
  }
  #pragma omp master taskloop simd linear(ref(j))
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp master taskloop simd linear(uval(j))
  for (int k = 0; k < argc; ++k) ++k;
  int v = 0;
  #pragma omp master taskloop simd linear(v:j)
  for (int k = 0; k < argc; ++k) { ++k; v += j; }
  #pragma omp master taskloop simd linear(i)
  for (int k = 0; k < argc; ++k) ++k;
  return 0;
}

namespace A {
double x;
#pragma omp threadprivate(x) // expected-note {{defined as threadprivate or thread local}}
}
namespace C {
using A::x;
}

void linear_modifiers(int argc) {
  int &f = argc;
  #pragma omp master taskloop simd linear(f)
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp master taskloop simd linear(val(f))
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp master taskloop simd linear(uval(f))
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp master taskloop simd linear(ref(f))
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp master taskloop simd linear(foo(f)) // expected-error {{expected one of 'ref', val' or 'uval' modifiers}}
  for (int k = 0; k < argc; ++k) ++k;
}

int f;
int main(int argc, char **argv) {
  double darr[100];
  // expected-note@+1 {{in instantiation of function template specialization 'test_template<-4, double, int>' requested here}}
  test_template<-4>(darr, 4);
  // expected-note@+1 {{in instantiation of function template specialization 'test_warn<0>' requested here}}
  test_warn<0>();

  S4 e(4); // expected-note {{'e' defined here}}
  S5 g(5); // expected-note {{'g' defined here}}
  int i, z;
  int &j = i;
  #pragma omp master taskloop simd linear(f) linear(f) // expected-error {{linear variable cannot be linear}} expected-note {{defined as linear}}
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp master taskloop simd linear // expected-error {{expected '(' after 'linear'}}
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp master taskloop simd linear ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp master taskloop simd linear () // expected-error {{expected expression}}
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp master taskloop simd linear (val // expected-error {{use of undeclared identifier 'val'}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp master taskloop simd linear (ref()) // expected-error {{expected expression}}
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp master taskloop simd linear (foo()) // expected-error {{expected expression}}
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp master taskloop simd linear (argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp master taskloop simd linear (argc, // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp master taskloop simd linear (argc > 0 ? argv[1] : argv[2]) // expected-error {{expected variable name}}
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp master taskloop simd linear (argc, z)
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp master taskloop simd linear (S1) // expected-error {{'S1' does not refer to a value}}
  for (int k = 0; k < argc; ++k) ++k;
  // expected-error@+2 {{linear variable with incomplete type 'S1'}}
  // expected-error@+1 {{argument of a linear clause should be of integral or pointer type, not 'S2'}}
  #pragma omp master taskloop simd linear(a, b)
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp master taskloop simd linear (argv[1]) // expected-error {{expected variable name}}
  for (int k = 0; k < argc; ++k) ++k;
  // expected-error@+2 {{argument of a linear clause should be of integral or pointer type, not 'S4'}}
  // expected-error@+1 {{argument of a linear clause should be of integral or pointer type, not 'S5'}}
  #pragma omp master taskloop simd linear(val(e, g))
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp master taskloop simd linear(h, C::x) // expected-error 2 {{threadprivate or thread local variable cannot be linear}}
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp parallel
  {
    int i;
    #pragma omp master taskloop simd linear(val(i))
    for (int k = 0; k < argc; ++k) ++k;
    #pragma omp master taskloop simd linear(uval(i) : 4) // expected-error {{variable of non-reference type 'int' can be used only with 'val' modifier, but used with 'uval'}}
    for (int k = 0; k < argc; ++k) { ++k; i += 4; }
  }
  #pragma omp master taskloop simd linear(ref(j))
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp master taskloop simd linear(i)
  for (int k = 0; k < argc; ++k) ++k;

  foomain<int,char>(argc,argv); // expected-note {{in instantiation of function template specialization 'foomain<int, char>' requested here}}
  return 0;
}


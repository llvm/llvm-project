// RUN: %clang_cc1 -verify -fopenmp %s -Wuninitialized
// RUN: %clang_cc1 -verify=expected,omp52 -fopenmp -fopenmp-version=52 -DOMP52 %s -Wuninitialized

// RUN: %clang_cc1 -verify -fopenmp-simd %s -Wuninitialized
// RUN: %clang_cc1 -verify=expected,omp52 -fopenmp-simd -fopenmp-version=52 -DOMP52 %s -Wuninitialized

extern int omp_default_mem_alloc;

void xxx(int argc) {
  int i, lin, step_sz; // expected-note {{initialize the variable 'lin' to silence this warning}} expected-note {{initialize the variable 'step_sz' to silence this warning}}
#pragma omp for linear(lin : step_sz) // expected-warning {{variable 'lin' is uninitialized when used here}} expected-warning {{variable 'step_sz' is uninitialized when used here}}
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
  #pragma omp for linear(B:bfoo())
  for (int i = 0; i < 10; ++i) ;
  // expected-error@+1 {{unexpected ':' in nested name specifier; did you mean '::'}}
  #pragma omp for linear(B::ib:B:bfoo())
  for (int i = 0; i < 10; ++i) ;
  // expected-error@+1 {{use of undeclared identifier 'ib'; did you mean 'B::ib'}}
  #pragma omp for linear(B:ib)
  for (int i = 0; i < 10; ++i) ;
  // expected-error@+1 {{unexpected ':' in nested name specifier; did you mean '::'?}}
  #pragma omp for linear(z:B:ib)
  for (int i = 0; i < 10; ++i) ;
  #pragma omp for linear(B:B::bfoo())
  for (int i = 0; i < 10; ++i) ;
  #pragma omp for linear(X::x : ::z)
  for (int i = 0; i < 10; ++i) ;
  #pragma omp for linear(B,::z, X::x)
  for (int i = 0; i < 10; ++i) ;
  #pragma omp for linear(::z)
  for (int i = 0; i < 10; ++i) ;
  // expected-error@+1 {{expected variable name}}
  #pragma omp for linear(B::bfoo())
  for (int i = 0; i < 10; ++i) ;
  #pragma omp for linear(B::ib,B:C1+C2)
  for (int i = 0; i < 10; ++i) ;
}

template<int L, class T, class N> T test_template(T* arr, N num) {
  N i;
  T sum = (T)0;
  T ind2 = - num * L; // expected-note {{'ind2' defined here}}
  // expected-error@+1 {{argument of a linear clause should be of integral or pointer type}}
#pragma omp for linear(ind2:L)
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
  #pragma omp for linear(ind2:LEN)
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
  int i, k;
  int &j = i;
  #pragma omp for linear // expected-error {{expected '(' after 'linear'}}
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp for linear ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp for linear () // expected-error {{expected expression}}
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp for linear (argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp for linear (argc, // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp for linear (argc > 0 ? argv[1] : argv[2]) // expected-error {{expected variable name}}
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp for linear (argc : 5) allocate , allocate(, allocate(omp_default , allocate(omp_default_mem_alloc, allocate(omp_default_mem_alloc:, allocate(omp_default_mem_alloc: argc, allocate(omp_default_mem_alloc: argv), allocate(argv) // expected-error {{expected '(' after 'allocate'}} expected-error 2 {{expected expression}} expected-error 2 {{expected ')'}} expected-error {{use of undeclared identifier 'omp_default'}} expected-note 2 {{to match this '('}}
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp for linear (S1) // expected-error {{'S1' does not refer to a value}}
  for (int k = 0; k < argc; ++k) ++k;
  // expected-error@+2 {{linear variable with incomplete type 'S1'}}
  // expected-error@+1 {{argument of a linear clause should be of integral or pointer type, not 'S2'}}
  #pragma omp for linear (a, b:B::ib)
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp for linear (argv[1]) // expected-error {{expected variable name}}
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp for linear(e, g, k)
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp for linear(h) // expected-error {{threadprivate or thread local variable cannot be linear}}
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp for linear(i)
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp parallel
  {
    int v = 0;
    int i;
    #pragma omp for linear(v:i)
    for (int k = 0; k < argc; ++k) { i = k; v += i; }
  }
  #pragma omp for linear(j)
  for (int k = 0; k < argc; ++k) ++k;
  int v = 0;
  #pragma omp for linear(v:j)
  for (int k = 0; k < argc; ++k) { ++k; v += j; }
  #pragma omp for linear(i)
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp for linear(i) ordered(1) // expected-error {{'linear' clause cannot be specified along with 'ordered' clause with a parameter}}
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

int main(int argc, char **argv) {
  double darr[100];
  // expected-note@+1 {{in instantiation of function template specialization 'test_template<-4, double, int>' requested here}}
  test_template<-4>(darr, 4);
  // expected-note@+1 {{in instantiation of function template specialization 'test_warn<0>' requested here}}
  test_warn<0>();

  S4 e(4); // expected-note {{'e' defined here}}
  S5 g(5); // expected-note {{'g' defined here}}
  int i, k;
  int &j = i;
  #pragma omp for linear // expected-error {{expected '(' after 'linear'}}
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp for linear ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp for linear () // expected-error {{expected expression}}
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp for linear (argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp for linear (argc, // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp for linear (argc > 0 ? argv[1] : argv[2]) // expected-error {{expected variable name}}
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp for linear (argc)
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp for linear (S1) // expected-error {{'S1' does not refer to a value}}
  for (int k = 0; k < argc; ++k) ++k;
  // expected-error@+2 {{linear variable with incomplete type 'S1'}}
  // expected-error@+1 {{argument of a linear clause should be of integral or pointer type, not 'S2'}}
  #pragma omp for linear(a, b, k)
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp for linear (argv[1]) // expected-error {{expected variable name}}
  for (int k = 0; k < argc; ++k) ++k;
  // expected-error@+2 {{argument of a linear clause should be of integral or pointer type, not 'S4'}}
  // expected-error@+1 {{argument of a linear clause should be of integral or pointer type, not 'S5'}}
  #pragma omp for linear(e, g)
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp for linear(h, C::x) // expected-error 2 {{threadprivate or thread local variable cannot be linear}}
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp parallel
  {
    int i;
    #pragma omp for linear(i)
    for (int k = 0; k < argc; ++k) ++k;
    #pragma omp for linear(val(i)) // omp52-error {{old syntax 'linear-modifier(list)' on 'linear' clause was deprecated, use new syntax 'linear(list: [linear-modifier,] step(step-size))'}}
    for (int k = 0; k < argc; ++k) ++k;
#ifdef OMP52
    #pragma omp for linear(i : step(4))
#else
    #pragma omp for linear(i : 4)
#endif
    for (int k = 0; k < argc; ++k) { ++k; i += 4; }
  }
#ifdef OMP52
  #pragma omp for linear(j: step() //omp52-error 2 {{expected expression}} omp52-error{{expected ')'}} omp52-note{{to match this '('}}
#else
  #pragma omp for linear(j)
#endif
  for (int k = 0; k < argc; ++k) ++k;
#ifdef OMP52
  #pragma omp for linear(i: step(1), val)
#else
  #pragma omp for linear(i)
#endif
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp for linear(i) ordered(1) // expected-error {{'linear' clause cannot be specified along with 'ordered' clause with a parameter}}
  for (int k = 0; k < argc; ++k) ++k;
#ifdef OMP52
  #pragma omp for linear(j: step()) // omp52-error 2 {{expected expression}}
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp for linear(j: step(1), step(2)) // omp52-error {{multiple 'step size' found in linear clause}}
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp for linear(j: val, val) // omp52-error {{multiple 'linear modifier' found in linear clause}}
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp for linear(j: pval) // omp52-error{{use of undeclared identifier 'pval'}}
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp for linear(i: val, step(2 // omp52-error 3 {{expected ')'}} omp52-note 2 {{to match this '('}}
  for (int k = 0; k < argc; ++k) ++k;
#endif

  foomain<int,char>(argc,argv); // expected-note {{n instantiation of function template specialization 'foomain<int, char>' requested here}}
  return 0;
}


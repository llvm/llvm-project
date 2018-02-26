// RUN: %clang_cc1 -verify -fopenmp %s

// RUN: %clang_cc1 -verify -fopenmp-simd %s

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
  #pragma omp for simd linear(B:bfoo())
  for (int i = 0; i < 10; ++i) ;
  // expected-error@+1 {{unexpected ':' in nested name specifier; did you mean '::'}}
  #pragma omp for simd linear(B::ib:B:bfoo())
  for (int i = 0; i < 10; ++i) ;
  // expected-error@+1 {{use of undeclared identifier 'ib'; did you mean 'B::ib'}}
  #pragma omp for simd linear(B:ib)
  for (int i = 0; i < 10; ++i) ;
  // expected-error@+1 {{unexpected ':' in nested name specifier; did you mean '::'?}}
  #pragma omp for simd linear(z:B:ib)
  for (int i = 0; i < 10; ++i) ;
  #pragma omp for simd linear(B:B::bfoo())
  for (int i = 0; i < 10; ++i) ;
  #pragma omp for simd linear(X::x : ::z)
  for (int i = 0; i < 10; ++i) ;
  #pragma omp for simd linear(B,::z, X::x)
  for (int i = 0; i < 10; ++i) ;
  #pragma omp for simd linear(::z)
  for (int i = 0; i < 10; ++i) ;
  // expected-error@+1 {{expected variable name}}
  #pragma omp for simd linear(B::bfoo())
  for (int i = 0; i < 10; ++i) ;
  #pragma omp for simd linear(B::ib,B:C1+C2)
  for (int i = 0; i < 10; ++i) ;
}

template<int L, class T, class N> T test_template(T* arr, N num) {
  N i;
  T sum = (T)0;
  T ind2 = - num * L; // expected-note {{'ind2' defined here}}
  // expected-error@+1 {{argument of a linear clause should be of integral or pointer type}}
#pragma omp for simd linear(ind2:L)
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
  #pragma omp for simd linear(ind2:LEN)
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
  int i;
  int &j = i;
  #pragma omp for simd linear // expected-error {{expected '(' after 'linear'}}
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp for simd linear ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp for simd linear () // expected-error {{expected expression}}
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp for simd linear (argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp for simd linear (argc, // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp for simd linear (argc > 0 ? argv[1] : argv[2]) // expected-error {{expected variable name}}
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp for simd linear (argc : 5)
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp for simd linear (S1) // expected-error {{'S1' does not refer to a value}}
  for (int k = 0; k < argc; ++k) ++k;
  // expected-error@+2 {{linear variable with incomplete type 'S1'}}
  // expected-error@+1 {{const-qualified variable cannot be linear}}
  #pragma omp for simd linear (a, b:B::ib)
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp for simd linear (argv[1]) // expected-error {{expected variable name}}
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp for simd linear(e, g)
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp for simd linear(h) // expected-error {{threadprivate or thread local variable cannot be linear}}
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp for simd linear(i)
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp parallel
  {
    int v = 0;
    int i;
    #pragma omp for simd linear(v:i)
    for (int k = 0; k < argc; ++k) { i = k; v += i; }
  }
  #pragma omp for simd linear(j)
  for (int k = 0; k < argc; ++k) ++k;
  int v = 0;
  #pragma omp for simd linear(v:j)
  for (int k = 0; k < argc; ++k) { ++k; v += j; }
  #pragma omp for simd linear(i)
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
  int i;
  int &j = i;
  #pragma omp for simd linear // expected-error {{expected '(' after 'linear'}}
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp for simd linear ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp for simd linear () // expected-error {{expected expression}}
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp for simd linear (argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp for simd linear (argc, // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp for simd linear (argc > 0 ? argv[1] : argv[2]) // expected-error {{expected variable name}}
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp for simd linear (argc)
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp for simd linear (S1) // expected-error {{'S1' does not refer to a value}}
  for (int k = 0; k < argc; ++k) ++k;
  // expected-error@+2 {{linear variable with incomplete type 'S1'}}
  // expected-error@+1 {{const-qualified variable cannot be linear}}
  #pragma omp for simd linear (a, b) 
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp for simd linear (argv[1]) // expected-error {{expected variable name}}
  for (int k = 0; k < argc; ++k) ++k;
  // expected-error@+2 {{argument of a linear clause should be of integral or pointer type, not 'S4'}}
  // expected-error@+1 {{argument of a linear clause should be of integral or pointer type, not 'S5'}}
  #pragma omp for simd linear(e, g)
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp for simd linear(h, C::x) // expected-error 2 {{threadprivate or thread local variable cannot be linear}}
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp parallel
  {
    int i;
    #pragma omp for simd linear(i)
    for (int k = 0; k < argc; ++k) ++k;
    #pragma omp for simd linear(i : 4)
    for (int k = 0; k < argc; ++k) { ++k; i += 4; }
  }
  #pragma omp for simd linear(j)
  for (int k = 0; k < argc; ++k) ++k;
  #pragma omp for simd linear(i)
  for (int k = 0; k < argc; ++k) ++k;

  foomain<int,char>(argc,argv); // expected-note {{in instantiation of function template specialization 'foomain<int, char>' requested here}}
  return 0;
}


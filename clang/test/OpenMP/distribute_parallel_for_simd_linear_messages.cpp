// RUN: %clang_cc1 -verify -fopenmp %s

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

// expected-error@+3 {{only loop iteration variables are allowed in 'linear' clause in distribute directives}}
#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd linear(B:bfoo())
  for (int i = 0; i < 10; ++i) ;

// expected-error@+3 {{only loop iteration variables are allowed in 'linear' clause in distribute directives}}
#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd linear(B::ib:B:bfoo()) // expected-error {{unexpected ':' in nested name specifier; did you mean '::'}}
  for (int i = 0; i < 10; ++i) ;

// expected-error@+3 {{only loop iteration variables are allowed in 'linear' clause in distribute directives}}
#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd linear(B:ib) // expected-error {{use of undeclared identifier 'ib'; did you mean 'B::ib'}}
  for (int i = 0; i < 10; ++i) ;

// expected-error@+3 {{only loop iteration variables are allowed in 'linear' clause in distribute directives}}
#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd linear(z:B:ib) // expected-error {{unexpected ':' in nested name specifier; did you mean '::'?}}
  for (int i = 0; i < 10; ++i) ;

// expected-error@+3 {{only loop iteration variables are allowed in 'linear' clause in distribute directives}}
#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd linear(B:B::bfoo())
  for (int i = 0; i < 10; ++i) ;

// expected-error@+3 {{only loop iteration variables are allowed in 'linear' clause in distribute directives}}
#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd linear(X::x : ::z)
  for (int i = 0; i < 10; ++i) ;

// expected-error@+3 3 {{only loop iteration variables are allowed in 'linear' clause in distribute directives}}
#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd linear(B,::z, X::x)
  for (int i = 0; i < 10; ++i) ;

// expected-error@+3 {{only loop iteration variables are allowed in 'linear' clause in distribute directives}}
#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd linear(::z)
  for (int i = 0; i < 10; ++i) ;

#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd linear(B::bfoo()) // expected-error {{expected variable name}}
  for (int i = 0; i < 10; ++i) ;

// expected-error@+3 2 {{only loop iteration variables are allowed in 'linear' clause in distribute directives}}
#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd linear(B::ib,B:C1+C2)
  for (int i = 0; i < 10; ++i) ;
}

template<int L, class T, class N> T test_template(T* arr, N num) {
  N i;
  T sum = (T)0;
  T ind2 = - num * L; // expected-note {{'ind2' defined here}}

#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd linear(ind2:L) // expected-error {{argument of a linear clause should be of integral or pointer type}}
  for (i = 0; i < num; ++i) {
    T cur = arr[(int)ind2];
    ind2 += L;
    sum += cur;
  }
  return T();
}

template<int LEN> int test_warn() {
  int ind2 = 0;
  #pragma omp target
  #pragma omp teams
  #pragma omp parallel for simd linear(ind2:LEN) // expected-warning {{zero linear step (ind2 should probably be const)}}
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

#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd linear // expected-error {{expected '(' after 'linear'}}
  for (int k = 0; k < argc; ++k) ++k;

#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd linear ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int k = 0; k < argc; ++k) ++k;

#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd linear () // expected-error {{expected expression}}
  for (int k = 0; k < argc; ++k) ++k;

// expected-error@+3 {{only loop iteration variables are allowed in 'linear' clause in distribute directives}}
#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd linear (argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int k = 0; k < argc; ++k) ++k;

// expected-error@+3 {{only loop iteration variables are allowed in 'linear' clause in distribute directives}}
#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd linear (argc, // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int k = 0; k < argc; ++k) ++k;

#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd linear (argc > 0 ? argv[1] : argv[2]) // expected-error {{expected variable name}}
  for (int k = 0; k < argc; ++k) ++k;

// expected-error@+3 {{only loop iteration variables are allowed in 'linear' clause in distribute directives}}
#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd linear (argc : 5)
  for (int k = 0; k < argc; ++k) ++k;

#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd linear (S1) // expected-error {{'S1' does not refer to a value}}
  for (int k = 0; k < argc; ++k) ++k;

#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd linear (a, b:B::ib) // expected-error {{linear variable with incomplete type 'S1'}} expected-error {{const-qualified variable cannot be linear}}
  for (int k = 0; k < argc; ++k) ++k;

#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd linear (argv[1]) // expected-error {{expected variable name}}
  for (int k = 0; k < argc; ++k) ++k;

// expected-error@+3 2 {{only loop iteration variables are allowed in 'linear' clause in distribute directives}}
#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd linear(e, g)
  for (int k = 0; k < argc; ++k) ++k;

#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd linear(h) // expected-error {{threadprivate or thread local variable cannot be linear}}
  for (int k = 0; k < argc; ++k) ++k;

// expected-error@+3 {{only loop iteration variables are allowed in 'linear' clause in distribute directives}}
#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd linear(i)
  for (int k = 0; k < argc; ++k) ++k;

  #pragma omp parallel
  {
    int v = 0;
    int i;
    int k;
    #pragma omp target
    #pragma omp teams
    #pragma omp distribute parallel for simd linear(k:i)
    for (k = 0; k < argc; ++k) { i = k; v += i; }
  }

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

#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd linear // expected-error {{expected '(' after 'linear'}}
  for (int k = 0; k < argc; ++k) ++k;

#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd linear ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int k = 0; k < argc; ++k) ++k;

#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd linear () // expected-error {{expected expression}}
  for (int k = 0; k < argc; ++k) ++k;

// expected-error@+3 {{only loop iteration variables are allowed in 'linear' clause in distribute directives}}
#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd linear (argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int k = 0; k < argc; ++k) ++k;

// expected-error@+3 {{only loop iteration variables are allowed in 'linear' clause in distribute directives}}
#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd linear (argc, // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int k = 0; k < argc; ++k) ++k;

#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd linear (argc > 0 ? argv[1] : argv[2]) // expected-error {{expected variable name}}
  for (int k = 0; k < argc; ++k) ++k;

// expected-error@+3 {{only loop iteration variables are allowed in 'linear' clause in distribute directives}}
#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd linear (argc)
  for (int k = 0; k < argc; ++k) ++k;

#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd linear (S1) // expected-error {{'S1' does not refer to a value}}
  for (int k = 0; k < argc; ++k) ++k;


#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd linear (a, b) // expected-error {{linear variable with incomplete type 'S1'}} expected-error {{const-qualified variable cannot be linear}}
  for (int k = 0; k < argc; ++k) ++k;

#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd linear (argv[1]) // expected-error {{expected variable name}}
  for (int k = 0; k < argc; ++k) ++k;

#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd linear(e, g) // expected-error {{argument of a linear clause should be of integral or pointer type, not 'S4'}} expected-error {{argument of a linear clause should be of integral or pointer type, not 'S5'}}
  for (int k = 0; k < argc; ++k) ++k;

#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd linear(h, C::x) // expected-error 2 {{threadprivate or thread local variable cannot be linear}}
  for (int k = 0; k < argc; ++k) ++k;

  #pragma omp parallel
  {
    int i;
    #pragma omp target
    #pragma omp teams
    #pragma omp distribute parallel for simd linear(i)
      for (i = 0; i < argc; ++i) ++i;

    #pragma omp target
    #pragma omp teams
    #pragma omp distribute parallel for simd linear(i : 4)
      for (i = 0; i < argc; ++i) { ++i; i += 4; }
  }

  foomain<int,char>(argc,argv); // expected-note {{in instantiation of function template specialization 'foomain<int, char>' requested here}}
  return 0;
}


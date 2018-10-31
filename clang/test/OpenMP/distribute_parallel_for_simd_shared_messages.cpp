// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 %s

// RUN: %clang_cc1 -verify -fopenmp-simd -ferror-limit 100 %s


struct S1; // expected-note 2 {{declared here}}
extern S1 a;
class S2 {
  mutable int a;
public:
  S2():a(0) { }
  S2(S2 &s2):a(s2.a) { }
};
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

S3 h;
#pragma omp threadprivate(h) // expected-note 2 {{defined as threadprivate or thread local}}

namespace A {
double x;
#pragma omp threadprivate(x) // expected-note 2 {{defined as threadprivate or thread local}}
}
namespace B {
using A::x;
}

template <class T, typename S, int N>
T tmain(T argc, S **argv) {
  const int d = 5;
  const int da[5] = { 0 };
  S4 e(4);
  S5 g(5);
  int i;
  int &j = i;
  int acc = 0;
  int n = 1000;
  
#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd shared // expected-error {{expected '(' after 'shared'}}
  for(int k = 0 ; k < n ; k++) {
    acc++;
  }

#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd shared ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for(int k = 0 ; k < n ; k++) {
    acc++;
  }

#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd shared () // expected-error {{expected expression}}
  for(int k = 0 ; k < n ; k++) {
    acc++;
  }

#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd shared (argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for(int k = 0 ; k < n ; k++) {
    acc++;
  }

#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd shared (argc, // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for(int k = 0 ; k < n ; k++) {
    acc++;
  }

#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd shared (argc > 0 ? argv[1] : argv[2]) // expected-error {{expected variable name}}
  for(int k = 0 ; k < n ; k++) {
    acc++;
  }

#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd shared (argc)
  for(int k = 0 ; k < n ; k++) {
    acc++;
  }

#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd shared (S1) // expected-error {{'S1' does not refer to a value}}
  for(int k = 0 ; k < n ; k++) {
    acc++;
  }

#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd shared (a, b, c, d, f) // expected-error {{incomplete type 'S1' where a complete type is required}} expected-warning {{Non-trivial type 'const S2' is mapped, only trivial types are guaranteed to be mapped correctly}} expected-warning {{Non-trivial type 'const S3' is mapped, only trivial types are guaranteed to be mapped correctly}}
  for(int k = 0 ; k < n ; k++) {
    acc++;
  }

#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd shared (argv[1]) // expected-error {{expected variable name}}
  for(int k = 0 ; k < n ; k++) {
    acc++;
  }

#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd shared(ba) // expected-warning {{Non-trivial type 'const S2 [5]' is mapped, only trivial types are guaranteed to be mapped correctly}}
  for(int k = 0 ; k < n ; k++) {
    acc++;
  }

#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd shared(ca) // expected-warning {{Non-trivial type 'const S3 [5]' is mapped, only trivial types are guaranteed to be mapped correctly}}
  for(int k = 0 ; k < n ; k++) {
    acc++;
  }

#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd shared(da)
  for(int k = 0 ; k < n ; k++) {
    acc++;
  }

#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd shared(e, g) // expected-warning {{Non-trivial type 'S4' is mapped, only trivial types are guaranteed to be mapped correctly}} expected-warning {{Non-trivial type 'S5' is mapped, only trivial types are guaranteed to be mapped correctly}}
  for(int k = 0 ; k < n ; k++) {
    acc++;
  }

#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd shared(h, B::x) // expected-error 2 {{threadprivate or thread local variable cannot be shared}}
  for(int k = 0 ; k < n ; k++) {
    acc++;
  }

#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd private(i), shared(i) // expected-error {{private variable cannot be shared}} expected-note {{defined as private}}
  for(int k = 0 ; k < n ; k++) {
    acc++;
  }

#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd firstprivate(i), shared(i) // expected-error {{firstprivate variable cannot be shared}} expected-note {{defined as firstprivate}}
  for(int k = 0 ; k < n ; k++) {
    acc++;
  }

#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd private(i)
  for(int k = 0 ; k < n ; k++) {
    acc++;
  }

#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd shared(i)
  for(int k = 0 ; k < n ; k++) {
    acc++;
  }

#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd shared(j)
  for(int k = 0 ; k < n ; k++) {
    acc++;
  }

#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd firstprivate(i)
  for(int k = 0 ; k < n ; k++) {
    acc++;
  }

#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd shared(i)
  for(int k = 0 ; k < n ; k++) {
    acc++;
  }

#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd shared(j)
  for(int k = 0 ; k < n ; k++) {
    acc++;
  }
return T();
}


int main(int argc, char **argv) {
  const int d = 5;
  const int da[5] = { 0 };
  S4 e(4);
  S5 g(5);
  int i;
  int &j = i;
  int acc = 0;
  int n = argc;

#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd shared // expected-error {{expected '(' after 'shared'}}
  for(int k = 0 ; k < n ; k++) {
    acc++;
  }

#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd shared ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for(int k = 0 ; k < n ; k++) {
    acc++;
  }

#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd shared () // expected-error {{expected expression}}
  for(int k = 0 ; k < n ; k++) {
    acc++;
  }

#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd shared (argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for(int k = 0 ; k < n ; k++) {
    acc++;
  }

#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd shared (argc, // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for(int k = 0 ; k < n ; k++) {
    acc++;
  }

#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd shared (argc > 0 ? argv[1] : argv[2]) // expected-error {{expected variable name}}
  for(int k = 0 ; k < n ; k++) {
    acc++;
  }

#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd shared (argc)
  for(int k = 0 ; k < n ; k++) {
    acc++;
  }

#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd shared (S1) // expected-error {{'S1' does not refer to a value}}
  for(int k = 0 ; k < n ; k++) {
    acc++;
  }

#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd shared (a, b, c, d, f) // expected-error {{incomplete type 'S1' where a complete type is required}} expected-warning {{Non-trivial type 'const S2' is mapped, only trivial types are guaranteed to be mapped correctly}} expected-warning {{Non-trivial type 'const S3' is mapped, only trivial types are guaranteed to be mapped correctly}}
  for(int k = 0 ; k < n ; k++) {
    acc++;
  }

#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd shared (argv[1]) // expected-error {{expected variable name}}
  for(int k = 0 ; k < n ; k++) {
    acc++;
  }

#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd shared(ba) // expected-warning {{Non-trivial type 'const S2 [5]' is mapped, only trivial types are guaranteed to be mapped correctly}}
  for(int k = 0 ; k < n ; k++) {
    acc++;
  }

#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd shared(ca) // expected-warning {{Non-trivial type 'const S3 [5]' is mapped, only trivial types are guaranteed to be mapped correctly}}
  for(int k = 0 ; k < n ; k++) {
    acc++;
  }

#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd shared(da)
  for(int k = 0 ; k < n ; k++) {
    acc++;
  }

#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd shared(e, g) // expected-warning {{Non-trivial type 'S4' is mapped, only trivial types are guaranteed to be mapped correctly}} expected-warning {{Non-trivial type 'S5' is mapped, only trivial types are guaranteed to be mapped correctly}}
  for(int k = 0 ; k < n ; k++) {
    acc++;
  }

#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd shared(h, B::x) // expected-error 2 {{threadprivate or thread local variable cannot be shared}}
  for(int k = 0 ; k < n ; k++) {
    acc++;
  }

#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd private(i), shared(i) // expected-error {{private variable cannot be shared}} expected-note {{defined as private}}
  for(int k = 0 ; k < n ; k++) {
    acc++;
  }

#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd firstprivate(i), shared(i) // expected-error {{firstprivate variable cannot be shared}} expected-note {{defined as firstprivate}}
  for(int k = 0 ; k < n ; k++) {
    acc++;
  }

#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd private(i)
  for(int k = 0 ; k < n ; k++) {
    acc++;
  }

#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd shared(i)
  for(int k = 0 ; k < n ; k++) {
    acc++;
  }

#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd shared(j)
  for(int k = 0 ; k < n ; k++) {
    acc++;
  }

#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd firstprivate(i)
  for(int k = 0 ; k < n ; k++) {
    acc++;
  }

#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd shared(i)
  for(int k = 0 ; k < n ; k++) {
    acc++;
  }

#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd shared(j)
  for(int k = 0 ; k < n ; k++) {
    acc++;
  }

return tmain<int, char, 1000>(argc, argv); // expected-note {{in instantiation of function template specialization 'tmain<int, char, 1000>' requested here}}
}

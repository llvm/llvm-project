// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 -o - %s -Wuninitialized

// RUN: %clang_cc1 -verify -fopenmp-simd -ferror-limit 100 -o - %s -Wuninitialized

void foo() {
}

bool foobool(int argc) {
  return argc;
}

struct S1; // expected-note {{declared here}}
class S2 {
  mutable int a;

public:
  S2() : a(0) {}
  S2 &operator=(S2 &s2) { return *this; }
};
class S3 {
  int a;

public:
  S3() : a(0) {}
  S3 &operator=(S3 &s3) { return *this; }
};
class S4 {
  int a;
  S4();
  S4 &operator=(const S4 &s4); // expected-note {{implicitly declared private here}}

public:
  S4(int v) : a(v) {}
};
class S5 {
  int a;
  S5() : a(0) {}
  S5 &operator=(const S5 &s5) { return *this; } // expected-note {{implicitly declared private here}}

public:
  S5(int v) : a(v) {}
};
template <class T>
class ST {
public:
  static T s;
};

S2 k;
S3 h;
S4 l(3);
S5 m(4);
#pragma omp threadprivate(h, k, l, m)

namespace A {
double x;
#pragma omp threadprivate(x)
}
namespace B {
using A::x;
}

int main(int argc, char **argv) {
  int i;
#pragma omp parallel sections copyin // expected-error {{expected '(' after 'copyin'}}
  {
    foo();
  }
#pragma omp parallel sections copyin( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  {
    foo();
  }
#pragma omp parallel sections copyin() // expected-error {{expected expression}}
  {
    foo();
  }
#pragma omp parallel sections copyin(k // expected-error {{expected ')'}} expected-note {{to match this '('}}
  {
    foo();
  }
#pragma omp parallel sections copyin(h, // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  {
    foo();
  }
#pragma omp parallel sections copyin(argc > 0 ? argv[1] : argv[2]) // expected-error {{expected variable name}}
  {
    foo();
  }
#pragma omp parallel sections copyin(l) // expected-error {{'operator=' is a private member of 'S4'}}
  {
    foo();
  }
#pragma omp parallel sections copyin(S1) // expected-error {{'S1' does not refer to a value}}
  {
    foo();
  }
#pragma omp parallel sections copyin(argv[1]) // expected-error {{expected variable name}}
  {
    foo();
  }
#pragma omp parallel sections copyin(i) // expected-error {{copyin variable must be threadprivate}}
  {
    foo();
  }
#pragma omp parallel sections copyin(m) // expected-error {{'operator=' is a private member of 'S5'}}
  {
    foo();
  }
#pragma omp parallel sections copyin(ST < int > ::s, B::x) // expected-error {{copyin variable must be threadprivate}}
  {
    foo();
  }

  return 0;
}

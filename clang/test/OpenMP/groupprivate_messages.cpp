// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -verify -fopenmp -fopenmp-version=60 -ferror-limit 100 -emit-llvm -o - %s
// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -verify -fopenmp -fopenmp-version=60 -ferror-limit 100 -emit-llvm -o - %s

// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -verify -fopenmp-simd -fopenmp-version=60 -ferror-limit 100 -emit-llvm -o - %s
// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -verify -fopenmp-simd -fopenmp-version=60 -ferror-limit 100 -emit-llvm -o - %s

#pragma omp groupprivate // expected-error {{expected '(' after 'groupprivate'}}
#pragma omp groupprivate( // expected-error {{expected identifier}} expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma omp groupprivate() // expected-error {{expected identifier}}
#pragma omp groupprivate(1) // expected-error {{expected unqualified-id}}
struct CompleteSt{
 int a;
};

struct CompleteSt1{
#pragma omp groupprivate(1) // expected-error {{expected unqualified-id}}
 int a;
} d; // expected-note {{'d' defined here}}

int a; // expected-note {{'a' defined here}}

#pragma omp groupprivate(a) allocate(a) // expected-warning {{extra tokens at the end of '#pragma omp groupprivate' are ignored}}
#pragma omp groupprivate(u) // expected-error {{use of undeclared identifier 'u'}}
int foo() { // expected-note {{declared here}}
  static int l;
#pragma omp groupprivate(l)) // expected-warning {{extra tokens at the end of '#pragma omp groupprivate' are ignored}}
  return (a);
}

int x, y;
#pragma omp groupprivate(x)) // expected-warning {{extra tokens at the end of '#pragma omp groupprivate' are ignored}}
#pragma omp groupprivate(y)),
// expected-warning@-1 {{extra tokens at the end of '#pragma omp groupprivate' are ignored}}
#pragma omp groupprivate(d.a) // expected-error {{expected identifier}}
#pragma omp groupprivate((float)a) // expected-error {{expected unqualified-id}}
int foa; // expected-note {{'foa' declared here}}
#pragma omp groupprivate(faa) // expected-error {{use of undeclared identifier 'faa'; did you mean 'foa'?}}
#pragma omp groupprivate(foo) // expected-error {{'foo' is not a global variable, static local variable or static data member}}
#pragma omp groupprivate (int a=2) // expected-error {{expected unqualified-id}}

struct IncompleteSt; // expected-note {{forward declaration of 'IncompleteSt'}}

extern IncompleteSt e;
#pragma omp groupprivate (e) // expected-error {{groupprivate variable with incomplete type 'IncompleteSt'}}

int &f = a; // expected-note {{'f' defined here}}
#pragma omp groupprivate (f) // expected-error {{arguments of '#pragma omp groupprivate' cannot be of reference type 'int &'}}

class TestClass {
  private:
    int a; // expected-note {{declared here}}
    static int b; // expected-note {{'b' declared here}}
    TestClass() : a(0){}
  public:
    TestClass (int aaa) : a(aaa) {}
#pragma omp groupprivate (b, a) // expected-error {{'a' is not a global variable, static local variable or static data member}}
} g(10);
#pragma omp groupprivate (b) // expected-error {{use of undeclared identifier 'b'}}
#pragma omp groupprivate (TestClass::b) // expected-error {{'#pragma omp groupprivate' must appear in the scope of the 'TestClass::b' variable declaration}}

const int h = 12; // expected-note {{'h' defined here}}
const volatile int i = 10; // expected-note {{'i' defined here}}
// For groupprivate these have initializers -> groupprivate forbids variables with initializers.
#pragma omp groupprivate (h, i) // expected-error {{variable 'h' with initializer cannot appear in groupprivate directive}} expected-error {{variable 'i' with initializer cannot appear in groupprivate directive}}

template <class T>
class TempClass {
  private:
    T a;
    TempClass() : a(){}
  public:
    TempClass (T aaa) : a(aaa) {}
    static T s;
#pragma omp groupprivate (s)
};
#pragma omp groupprivate (s) // expected-error {{use of undeclared identifier 's'}}

int o; // expected-note {{candidate found by name lookup is 'o'}}
#pragma omp groupprivate (o)
namespace {
int o; // expected-note {{candidate found by name lookup is '(anonymous namespace)::o'}}
#pragma omp groupprivate (o)
}
#pragma omp groupprivate (o) // expected-error {{reference to 'o' is ambiguous}}

int main(int argc, char **argv) {

  int x, y = argc;
  static double d1;
  static double d2;
  static double d3; // expected-note {{'d3' defined here}}
  static double d4;

  d.a = a;
  d2++;
  ;
#pragma omp groupprivate(argc+y) // expected-error {{expected identifier}}
#pragma omp groupprivate(d2) // expected-error {{'#pragma omp groupprivate' must precede all references to variable 'd2'}}
#pragma omp groupprivate(d1)
  {
  ++a;d2=0;
#pragma omp groupprivate(d3) // expected-error {{'#pragma omp groupprivate' must appear in the scope of the 'd3' variable declaration}}
  }
#pragma omp groupprivate(d3)
label:
#pragma omp groupprivate(d4) // expected-error {{'#pragma omp groupprivate' cannot be an immediate substatement}}

#pragma omp groupprivate(a) // expected-error {{'#pragma omp groupprivate' must appear in the scope of the 'a' variable declaration}}
  return (y);
#pragma omp groupprivate(d) // expected-error {{'#pragma omp groupprivate' must appear in the scope of the 'd' variable declaration}}
}

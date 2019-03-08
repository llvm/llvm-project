// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -verify -fopenmp -fnoopenmp-use-tls -ferror-limit 100 -emit-llvm -o - %s
// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -verify -fopenmp -ferror-limit 100 -emit-llvm -o - %s

// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -verify -fopenmp-simd -fnoopenmp-use-tls -ferror-limit 100 -emit-llvm -o - %s
// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -verify -fopenmp-simd -ferror-limit 100 -emit-llvm -o - %s

#pragma omp allocate    // expected-error {{expected '(' after 'allocate'}}
#pragma omp allocate(   // expected-error {{expected identifier}} expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma omp allocate()  // expected-error {{expected identifier}}
#pragma omp allocate(1) // expected-error {{expected unqualified-id}}
struct CompleteSt {
  int a;
};

struct CompleteSt1 {
#pragma omp allocate(1) // expected-error {{expected unqualified-id}}
  int a;
} d; // expected-note {{'d' defined here}}

int a; // expected-note {{'a' defined here}}

#pragma omp allocate(a)
#pragma omp allocate(u) // expected-error {{use of undeclared identifier 'u'}}
#pragma omp allocate(d, a)
int foo() { // expected-note {{declared here}}
  static int l;
#pragma omp allocate(l)) // expected-warning {{extra tokens at the end of '#pragma omp allocate' are ignored}}
  return (a);
}

#pragma omp allocate(a)(
// expected-warning@-1 {{extra tokens at the end of '#pragma omp allocate' are ignored}}
#pragma omp allocate(a)[  // expected-warning {{extra tokens at the end of '#pragma omp allocate' are ignored}}
#pragma omp allocate(a) { // expected-warning {{extra tokens at the end of '#pragma omp allocate' are ignored}}
#pragma omp allocate(a))  // expected-warning {{extra tokens at the end of '#pragma omp allocate' are ignored}}
#pragma omp allocate(a)]  // expected-warning {{extra tokens at the end of '#pragma omp allocate' are ignored}}
#pragma omp allocate(a) } // expected-warning {{extra tokens at the end of '#pragma omp allocate' are ignored}}
#pragma omp allocate a    // expected-error {{expected '(' after 'allocate'}}
#pragma omp allocate(d    // expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma omp allocate(d))  // expected-warning {{extra tokens at the end of '#pragma omp allocate' are ignored}}
int x, y;
#pragma omp allocate(x)) // expected-warning {{extra tokens at the end of '#pragma omp allocate' are ignored}}
#pragma omp allocate(y)),
// expected-warning@-1 {{extra tokens at the end of '#pragma omp allocate' are ignored}}
#pragma omp allocate(a, d)
#pragma omp allocate(d.a)       // expected-error {{expected identifier}}
#pragma omp allocate((float)a)  // expected-error {{expected unqualified-id}}
int foa;                        // expected-note {{'foa' declared here}}
#pragma omp allocate(faa)       // expected-error {{use of undeclared identifier 'faa'; did you mean 'foa'?}}
#pragma omp allocate(foo)       // expected-error {{'foo' is not a global variable, static local variable or static data member}}
#pragma omp allocate(int a = 2) // expected-error {{expected unqualified-id}}

struct IncompleteSt;

extern IncompleteSt e;
#pragma omp allocate(e)

int &f = a;
#pragma omp allocate(f)

class TestClass {
private:
  int a;        // expected-note {{declared here}}
  static int b; // expected-note {{'b' declared here}}
  TestClass() : a(0) {}

public:
  TestClass(int aaa) : a(aaa) {}
#pragma omp allocate(b, a) // expected-error {{'a' is not a global variable, static local variable or static data member}}
} g(10);
#pragma omp allocate(b)            // expected-error {{use of undeclared identifier 'b'}}
#pragma omp allocate(TestClass::b) // expected-error {{'#pragma omp allocate' must appear in the scope of the 'TestClass::b' variable declaration}}
#pragma omp allocate(g)

namespace ns {
int m;
#pragma omp allocate(m, m)
} // namespace ns
#pragma omp allocate(m) // expected-error {{use of undeclared identifier 'm'}}
#pragma omp allocate(ns::m)
#pragma omp allocate(ns \
                     : m) // expected-error {{unexpected ':' in nested name specifier; did you mean '::'?}}

const int h = 12;
const volatile int i = 10;
#pragma omp allocate(h, i)

template <class T>
class TempClass {
private:
  T a;
  TempClass() : a() {}

public:
  TempClass(T aaa) : a(aaa) {}
  static T s;
#pragma omp allocate(s)
};
#pragma omp allocate(s) // expected-error {{use of undeclared identifier 's'}}

static __thread int t;
#pragma omp allocate(t)

// Register "0" is currently an invalid register for global register variables.
// Use "esp" instead of "0".
// register int reg0 __asm__("0");
register int reg0 __asm__("esp");
#pragma omp allocate(reg0)

int o; // expected-note {{candidate found by name lookup is 'o'}}
#pragma omp allocate(o)
namespace {
int o; // expected-note {{candidate found by name lookup is '(anonymous namespace)::o'}}
#pragma omp allocate(o)
#pragma omp allocate(o)
} // namespace
#pragma omp allocate(o) // expected-error {{reference to 'o' is ambiguous}}
#pragma omp allocate(::o)

int main(int argc, char **argv) {

  int x, y = argc;
  static double d1;
  static double d2;
  static double d3; // expected-note {{'d3' defined here}}
  static double d4;
  static TestClass LocalClass(y);
#pragma omp allocate(LocalClass)

  d.a = a;
  d2++;
  ;
#pragma omp allocate(argc + y) // expected-error {{expected identifier}}
#pragma omp allocate(argc, y)
#pragma omp allocate(d2)
#pragma omp allocate(d1)
  {
    ++a;
    d2 = 0;
#pragma omp allocate(d3) // expected-error {{'#pragma omp allocate' must appear in the scope of the 'd3' variable declaration}}
  }
#pragma omp allocate(d3)
label:
#pragma omp allocate(d4) // expected-error {{'#pragma omp allocate' cannot be an immediate substatement}}

#pragma omp allocate(a) // expected-error {{'#pragma omp allocate' must appear in the scope of the 'a' variable declaration}}
  return (y);
#pragma omp allocate(d) // expected-error {{'#pragma omp allocate' must appear in the scope of the 'd' variable declaration}}
}

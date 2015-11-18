// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -verify -fopenmp -fnoopenmp-use-tls -ferror-limit 100 -emit-llvm -o - %s
// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -verify -fopenmp -ferror-limit 100 -emit-llvm -o - %s

#pragma omp threadprivate // expected-error {{expected '(' after 'threadprivate'}}
#pragma omp threadprivate( // expected-error {{expected identifier}} expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma omp threadprivate() // expected-error {{expected identifier}}
#pragma omp threadprivate(1) // expected-error {{expected unqualified-id}}
struct CompleteSt{
 int a;
};

struct CompleteSt1{
#pragma omp threadprivate(1) // expected-error {{expected unqualified-id}}
 int a;
} d; // expected-note {{'d' defined here}}

int a; // expected-note {{'a' defined here}}

#pragma omp threadprivate(a)
#pragma omp threadprivate(u) // expected-error {{use of undeclared identifier 'u'}}
#pragma omp threadprivate(d, a)
int foo() { // expected-note {{declared here}}
  static int l;
#pragma omp threadprivate(l)) // expected-warning {{extra tokens at the end of '#pragma omp threadprivate' are ignored}}
  return (a);
}

#pragma omp threadprivate (a) (
// expected-warning@-1 {{extra tokens at the end of '#pragma omp threadprivate' are ignored}}
#pragma omp threadprivate (a) [ // expected-warning {{extra tokens at the end of '#pragma omp threadprivate' are ignored}}
#pragma omp threadprivate (a) { // expected-warning {{extra tokens at the end of '#pragma omp threadprivate' are ignored}}
#pragma omp threadprivate (a) ) // expected-warning {{extra tokens at the end of '#pragma omp threadprivate' are ignored}}
#pragma omp threadprivate (a) ] // expected-warning {{extra tokens at the end of '#pragma omp threadprivate' are ignored}}
#pragma omp threadprivate (a) } // expected-warning {{extra tokens at the end of '#pragma omp threadprivate' are ignored}}
#pragma omp threadprivate a // expected-error {{expected '(' after 'threadprivate'}}
#pragma omp threadprivate(d // expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma omp threadprivate(d)) // expected-warning {{extra tokens at the end of '#pragma omp threadprivate' are ignored}}
int x, y;
#pragma omp threadprivate(x)) // expected-warning {{extra tokens at the end of '#pragma omp threadprivate' are ignored}}
#pragma omp threadprivate(y)),
// expected-warning@-1 {{extra tokens at the end of '#pragma omp threadprivate' are ignored}}
#pragma omp threadprivate(a,d)
#pragma omp threadprivate(d.a) // expected-error {{expected identifier}}
#pragma omp threadprivate((float)a) // expected-error {{expected unqualified-id}}
int foa; // expected-note {{'foa' declared here}}
#pragma omp threadprivate(faa) // expected-error {{use of undeclared identifier 'faa'; did you mean 'foa'?}}
#pragma omp threadprivate(foo) // expected-error {{'foo' is not a global variable, static local variable or static data member}}
#pragma omp threadprivate (int a=2) // expected-error {{expected unqualified-id}}

struct IncompleteSt; // expected-note {{forward declaration of 'IncompleteSt'}}

extern IncompleteSt e;
#pragma omp threadprivate (e) // expected-error {{threadprivate variable with incomplete type 'IncompleteSt'}}

int &f = a; // expected-note {{'f' defined here}}
#pragma omp threadprivate (f) // expected-error {{arguments of '#pragma omp threadprivate' cannot be of reference type 'int &'}}

class TestClass {
  private:
    int a; // expected-note {{declared here}}
    static int b; // expected-note {{'b' declared here}}
    TestClass() : a(0){}
  public:
    TestClass (int aaa) : a(aaa) {}
#pragma omp threadprivate (b, a) // expected-error {{'a' is not a global variable, static local variable or static data member}}
} g(10);
#pragma omp threadprivate (b) // expected-error {{use of undeclared identifier 'b'}}
#pragma omp threadprivate (TestClass::b) // expected-error {{'#pragma omp threadprivate' must appear in the scope of the 'TestClass::b' variable declaration}}
#pragma omp threadprivate (g)

namespace ns {
  int m;
#pragma omp threadprivate (m)
}
#pragma omp threadprivate (m) // expected-error {{use of undeclared identifier 'm'}}
#pragma omp threadprivate (ns::m)
#pragma omp threadprivate (ns:m) // expected-error {{unexpected ':' in nested name specifier; did you mean '::'?}}

const int h = 12;
const volatile int i = 10;
#pragma omp threadprivate (h, i)


template <class T>
class TempClass {
  private:
    T a;
    TempClass() : a(){}
  public:
    TempClass (T aaa) : a(aaa) {}
    static T s;
#pragma omp threadprivate (s)
};
#pragma omp threadprivate (s) // expected-error {{use of undeclared identifier 's'}}

static __thread int t; // expected-note {{'t' defined here}}
#pragma omp threadprivate (t) // expected-error {{variable 't' cannot be threadprivate because it is thread-local}}

// Register "0" is currently an invalid register for global register variables.
// Use "esp" instead of "0".
// register int reg0 __asm__("0");
register int reg0 __asm__("esp"); // expected-note {{'reg0' defined here}}
#pragma omp threadprivate (reg0) // expected-error {{variable 'reg0' cannot be threadprivate because it is a global named register variable}}

int o; // expected-note {{candidate found by name lookup is 'o'}}
#pragma omp threadprivate (o)
namespace {
int o; // expected-note {{candidate found by name lookup is '(anonymous namespace)::o'}}
#pragma omp threadprivate (o)
#pragma omp threadprivate (o)
}
#pragma omp threadprivate (o) // expected-error {{reference to 'o' is ambiguous}}
#pragma omp threadprivate (::o)

int main(int argc, char **argv) { // expected-note {{'argc' defined here}}

  int x, y = argc; // expected-note 2 {{'y' defined here}}
  static double d1;
  static double d2;
  static double d3; // expected-note {{'d3' defined here}}
  static TestClass LocalClass(y); // expected-error {{variable with local storage in initial value of threadprivate variable}}
#pragma omp threadprivate(LocalClass)

  d.a = a;
  d2++;
  ;
#pragma omp threadprivate(argc+y) // expected-error {{expected identifier}}
#pragma omp threadprivate(argc,y) // expected-error 2 {{arguments of '#pragma omp threadprivate' must have static storage duration}}
#pragma omp threadprivate(d2) // expected-error {{'#pragma omp threadprivate' must precede all references to variable 'd2'}}
#pragma omp threadprivate(d1)
  {
  ++a;d2=0;
#pragma omp threadprivate(d3) // expected-error {{'#pragma omp threadprivate' must appear in the scope of the 'd3' variable declaration}}
  }
#pragma omp threadprivate(d3)

#pragma omp threadprivate(a) // expected-error {{'#pragma omp threadprivate' must appear in the scope of the 'a' variable declaration}}
  return (y);
#pragma omp threadprivate(d) // expected-error {{'#pragma omp threadprivate' must appear in the scope of the 'd' variable declaration}}
}

// RUN: %clang_cc1 -verify -std=c++11 %s

namespace RedeclAliasTypedef {
  typedef int T;
  using T = int;
  using T = int;
  typedef T T;
  using T = T;
  typedef int T;
}

namespace IllegalTypeIds {
  using A = void(int n = 0); // expected-error {{default arguments can only be specified for parameters in a function declaration}}
  using B = inline void(int n); // expected-error {{type name does not allow function specifier}}
  using C = virtual void(int n); // expected-error {{type name does not allow function specifier}}
  using D = explicit void(int n); // expected-error {{type name does not allow function specifier}}
  using E = void(int n) throw(); // expected-error {{exception specifications are not allowed in type aliases}}
  using F = void(*)(int n) &&; // expected-error {{pointer to function type cannot have '&&' qualifier}}
  using G = __thread void(int n); // expected-error {{type name does not allow storage class to be specified}}
  using H = constexpr int; // expected-error {{type name does not allow constexpr specifier}}

  using Y = void(int n); // ok
  using Z = void(int n) &&; // ok
}

namespace IllegalSyntax {
  using ::T = void(int n); // expected-error {{name defined in alias declaration must be an identifier}}
  using operator int = void(int n); // expected-error {{name defined in alias declaration must be an identifier}}
  using typename U = void; // expected-error {{name defined in alias declaration must be an identifier}}
  using typename ::V = void(int n); // expected-error {{name defined in alias declaration must be an identifier}}
  using typename ::operator bool = void(int n); // expected-error {{name defined in alias declaration must be an identifier}}
}

namespace VariableLengthArrays {
  using T = int[42]; // ok

  int n = 32;       // expected-note {{declared here}}
  using T = int[n]; // expected-error {{variable length array declaration not allowed at file scope}} \
                       expected-warning {{variable length arrays in C++ are a Clang extension}} \
                       expected-note {{read of non-const variable 'n' is not allowed in a constant expression}}

  const int m = 42;
  using U = int[m];
  using U = int[42]; // expected-note {{previous definition}}
  using U = int; // expected-error {{type alias redefinition with different types ('int' vs 'int[42]')}}

  void f() {
    int n = 42; // expected-note {{declared here}}
    goto foo; // expected-error {{cannot jump}}
    using T = int[n]; // expected-note {{bypasses initialization of VLA type alias}} \
                         expected-warning {{variable length arrays in C++ are a Clang extension}} \
                         expected-note {{read of non-const variable 'n' is not allowed in a constant expression}}
  foo: ;
  }
}

namespace RedeclFunc {
  int f(int, char**);
  using T = int;
  T f(int, char **); // ok
}

namespace LookupFilter {
  namespace N { struct S; }
  using namespace N;
  using S = S*; // ok
}

namespace InFunctions {
  template<typename...T> void f0() {
    using U = T*; // expected-error {{declaration type contains unexpanded parameter pack 'T'}}
    U u;
  }
  template void f0<int, char>();

  void f1() {
    using T = int;
  }
  void f2() {
    using T = int[-1]; // expected-error {{array size is negative}}
  }

  template<typename...T> void f3() { // expected-note {{template parameter is declared here}}
    using T = int; // expected-error {{declaration of 'T' shadows template parameter}}
  }
}

namespace ClassNameRedecl {
  class C0 {
    using C0 = int; // expected-error {{member 'C0' has the same name as its class}}
  };
  class C1 {
    using C1 = C1; // expected-error {{member 'C1' has the same name as its class}}
  };
  class C2 {
    using C0 = C1; // ok
  };
  template<typename...T> class C3 {
    using f = T; // expected-error {{declaration type contains unexpanded parameter pack 'T'}}
  };
  template<typename T> class C4 { // expected-note {{template parameter is declared here}}
    using T = int; // expected-error {{declaration of 'T' shadows template parameter}}
  };
  class C5 {
    class c; // expected-note {{previous definition}}
    using c = int; // expected-error {{typedef redefinition with different types}}
    class d;
    using d = d; // ok
  };
  class C6 {
    class c { using C6 = int; }; // ok
  };
}

class CtorDtorName {
  using X = CtorDtorName;
  X(); // expected-error {{expected member name}}
  ~X(); // expected-error {{destructor cannot be declared using a type alias}}
};

namespace TagName {
  using S = struct { int n; };
  using T = class { int n; };
  using U = enum { a, b, c };
  using V = struct V { int n; };
}

namespace CWG1044 {
  using T = T; // expected-error {{unknown type name 'T'}}
}

namespace StdExample {
  template<typename T, typename U> struct pair;

  using handler_t = void (*)(int);
  extern handler_t ignore;
  extern void (*ignore)(int);
  // FIXME: we know we're parsing a type here; don't recover as if we were
  // using operator*.
  using cell = pair<void*, cell*>; // expected-error {{use of undeclared identifier 'cell'}} \
                                      expected-error {{expected expression}}
}

namespace Access {
  class C0 {
    using U = int; // expected-note {{declared private here}}
  };
  C0::U v; // expected-error {{'U' is a private member}}
  class C1 {
  public:
    using U = int;
  };
  C1::U w; // ok
}

namespace VoidArg {
  using V = void;
  V f(int); // ok
  V g(V); // ok (DR577)
}

// RUN: %clang_cc1 -verify -std=c++11 %s

namespace RedeclAliasTypedef {
  template<typename U> using T = int;
  template<typename U> using T = int;
  template<typename U> using T = T<U>;
}

namespace IllegalTypeIds {
  template<typename U> using A = void(int n = 0); // expected-error {{default arguments can only be specified for parameters in a function declaration}}
  template<typename U> using B = inline void(int n); // expected-error {{type name does not allow function specifier}}
  template<typename U> using C = virtual void(int n); // expected-error {{type name does not allow function specifier}}
  template<typename U> using D = explicit void(int n); // expected-error {{type name does not allow function specifier}}
  template<typename U> using E = void(int n) throw(); // expected-error {{exception specifications are not allowed in type aliases}}
  template<typename U> using F = void(*)(int n) &&; // expected-error {{pointer to function type cannot have '&&' qualifier}}
  template<typename U> using G = __thread void(int n); // expected-error {{type name does not allow storage class to be specified}}
  template<typename U> using H = constexpr int; // expected-error {{type name does not allow constexpr specifier}}

  template<typename U> using Y = void(int n); // ok
  template<typename U> using Z = void(int n) &&; // ok
}

namespace IllegalSyntax {
  template<typename Z> using ::T = void(int n); // expected-error {{name defined in alias declaration must be an identifier}}
  template<typename Z> using operator int = void(int n); // expected-error {{name defined in alias declaration must be an identifier}}
  template<typename Z> using typename U = void; // expected-error {{name defined in alias declaration must be an identifier}}
  template<typename Z> using typename ::V = void(int n); // expected-error {{name defined in alias declaration must be an identifier}}
  template<typename Z> using typename ::operator bool = void(int n); // expected-error {{name defined in alias declaration must be an identifier}}
}

namespace VariableLengthArrays {
  template<typename Z> using T = int[42]; // ok

  int n = 32;
  template<typename Z> using T = int[n]; // expected-error {{variable length array declaration not allowed at file scope}}

  const int m = 42;
  template<typename Z> using U = int[m]; // expected-note {{previous definition}}
  template<typename Z> using U = int[42]; // ok
  template<typename Z> using U = int; // expected-error {{type alias template redefinition with different types ('int' vs 'int [42]')}}
}

namespace RedeclFunc {
  int f(int, char**);
  template<typename Z> using T = int;
  T<char> f(int, char **); // ok
}

namespace LookupFilter {
  namespace N { template<typename U> using S = int; }
  using namespace N;
  template<typename U> using S = S<U>*; // ok
}

namespace InFunctions {
  template<typename...T> struct S0 {
    template<typename Z> using U = T*; // expected-error {{declaration type contains unexpanded parameter pack 'T'}}
    U<char> u;
  };

  template<typename Z> using T1 = int;
  template<typename Z> using T2 = int[-1]; // expected-error {{array size is negative}}
  template<typename...T> struct S3 { // expected-note {{template parameter is declared here}}
    template<typename Z> using T = int; // expected-error {{declaration of 'T' shadows template parameter}}
  };
  template<typename Z> using Z = Z;
}

namespace ClassNameRedecl {
  class C0 {
    // FIXME: this diagnostic is pretty poor
    template<typename U> using C0 = int; // expected-error {{name defined in alias declaration must be an identifier}}
  };
  class C1 {
    // FIXME: this diagnostic is pretty poor
    template<typename U> using C1 = C1; // expected-error {{name defined in alias declaration must be an identifier}}
  };
  class C2 {
    template<typename U> using C0 = C1; // ok
  };
  template<typename...T> class C3 {
    template<typename U> using f = T; // expected-error {{declaration type contains unexpanded parameter pack 'T'}}
  };
  template<typename T> class C4 { // expected-note {{template parameter is declared here}}
    template<typename U> using T = int; // expected-error {{declaration of 'T' shadows template parameter}}
  };
  class C5 {
    class c; // expected-note {{previous definition}}
    template<typename U> using c = int; // expected-error {{redefinition of 'c' as different kind of symbol}}
    class d; // expected-note {{previous definition}}
    template<typename U> using d = d; // expected-error {{redefinition of 'd' as different kind of symbol}}
  };
  class C6 {
    class c { template<typename U> using C6 = int; }; // ok
  };
}

class CtorDtorName {
  template<typename T> using X = CtorDtorName;
  X<int>(); // expected-error {{expected member name}}
  ~X<int>(); // expected-error {{destructor cannot be declared using a type alias}}
};

namespace TagName {
  template<typename Z> using S = struct { int n; }; // expected-error {{can not be defined}}
  template<typename Z> using T = class { int n; }; // expected-error {{can not be defined}}
  template<typename Z> using U = enum { a, b, c }; // expected-error {{can not be defined}}
  template<typename Z> using V = struct V { int n; }; // expected-error {{redefinition of 'V' as different kind of symbol}} \
                                                         expected-error {{'TagName::V' can not be defined in a type alias template}} \
                                                         expected-note {{previous definition is here}}
}

namespace StdExample {
  template<typename T, typename U> struct pair;

  template<typename T> using handler_t = void (*)(T);
  extern handler_t<int> ignore;
  extern void (*ignore)(int);
  // FIXME: we recover as if cell is an undeclared variable. the diagnostics are terrible!
  template<typename T> using cell = pair<T*, cell<T>*>; // expected-error {{use of undeclared identifier 'cell'}} \
                                                           expected-error {{'T' does not refer to a value}} \
                                                           expected-note {{declared here}} \
                                                           expected-error {{expected ';' after alias declaration}}
}

namespace Access {
  class C0 {
    template<typename Z> using U = int; // expected-note {{declared private here}}
  };
  C0::U<int> v; // expected-error {{'U' is a private member}}
  class C1 {
  public:
    template<typename Z> using U = int;
  };
  C1::U<int> w; // ok
}

namespace VoidArg {
  template<typename Z> using V = void;
  V<int> f(int); // ok
  V<char> g(V<double>); // expected-error {{empty parameter list defined with a type alias of 'void' not allowed}}
}

namespace Curried {
  template<typename T, typename U> struct S;
  template<typename T> template<typename U> using SS = S<T, U>; // expected-error {{extraneous template parameter list in alias template declaration}}
}

// PR12647
namespace SFINAE {
  template<bool> struct enable_if; // expected-note 2{{here}}
  template<> struct enable_if<true> { using type = void; };

  template<typename T> struct is_enum { static constexpr bool value = __is_enum(T); };

  template<typename T> using EnableIf = typename enable_if<T::value>::type; // expected-error {{undefined template}}
  template<typename T> using DisableIf = typename enable_if<!T::value>::type; // expected-error {{undefined template}}

  template<typename T> EnableIf<is_enum<T>> f();
  template<typename T> DisableIf<is_enum<T>> f();

  enum E { e };

  int main() {
    f<int>();
    f<E>();
  }

  template<typename T, typename U = EnableIf<is_enum<T>>> struct fail1 {}; // expected-note {{here}}
  template<typename T> struct fail2 : DisableIf<is_enum<T>> {}; // expected-note {{here}}

  fail1<int> f1; // expected-note {{here}}
  fail2<E> f2; // expected-note {{here}}
}

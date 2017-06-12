// RUN: %clang_cc1 -fsyntax-only -std=c++98 -verify -fblocks %s 
namespace A {
  struct C {
    static int cx;

    static int cx2;

    static int Ag1();
    static int Ag2();
  };
  int ax; // expected-note {{'ax' declared here}}
  void Af();
}

A:: ; // expected-error {{expected unqualified-id}}
::A::ax::undef ex3; // expected-error {{'ax' is not a class, namespace, or enumeration}}
A::undef1::undef2 ex4; // expected-error {{no member named 'undef1'}}

int A::C::Ag1() { return 0; }

static int A::C::Ag2() { return 0; } // expected-error{{'static' can}}

int A::C::cx = 17;


static int A::C::cx2 = 17; // expected-error{{'static' can}}

class C2 {
  void m(); // expected-note{{member declaration does not match because it is not const qualified}}

  void f(const int& parm); // expected-note{{type of 1st parameter of member declaration does not match definition ('const int &' vs 'int')}}
  void f(int) const; // expected-note{{member declaration does not match because it is const qualified}}
  void f(float);

  int x;
};

void C2::m() const { } // expected-error{{out-of-line definition of 'm' does not match any declaration in 'C2'}}

void C2::f(int) { } // expected-error{{out-of-line definition of 'f' does not match any declaration in 'C2'}}

void C2::m() {
  x = 0;
}

namespace B {
  void ::A::Af() {} // expected-error {{cannot define or redeclare 'Af' here because namespace 'B' does not enclose namespace 'A'}}
}

void f1() {
  void A::Af(); // expected-error {{definition or redeclaration of 'Af' not allowed inside a function}}
  void (^x)() = ^{ void A::Af(); }; // expected-error {{definition or redeclaration of 'Af' not allowed inside a block}}
}

void f2() {
  A:: ; // expected-error {{expected unqualified-id}}
  A::C::undef = 0; // expected-error {{no member named 'undef'}}
  ::A::C::cx = 0;
  int x = ::A::ax = A::C::cx;
  x = sizeof(A::C);
  x = sizeof(::A::C::cx);
}

A::C c1;
struct A::C c2;
struct S : public A::C {};
struct A::undef; // expected-error {{no struct named 'undef' in namespace 'A'}}

namespace A2 {
  typedef int INT;
  struct RC;
  struct CC {
    struct NC;
  };
}

struct A2::RC {
  INT x;
};

struct A2::CC::NC {
  void m() {}
};

void f3() {
  N::x = 0; // expected-error {{use of undeclared identifier 'N'}}
  // FIXME: Consider including the kind of entity that 'N' is ("variable 'N'
  // declared here", "template 'X' declared here", etc) to help explain what it
  // is if it's 'not a class, namespace, or scoped enumeration'.
  int N; // expected-note {{'N' declared here}}
  N::x = 0; // expected-error {{'N' is not a class, namespace, or enumeration}}
  { int A;           A::ax = 0; }
  { typedef int A;   A::ax = 0; } // expected-error{{'A' (aka 'int') is not a class, namespace, or enumeration}}
  { typedef A::C A;  A::ax = 0; } // expected-error {{no member named 'ax'}}
  { typedef A::C A;  A::cx = 0; }
}

// make sure the following doesn't hit any asserts
void f4(undef::C); // expected-error {{use of undeclared identifier 'undef'}}

typedef void C2::f5(int); // expected-error{{typedef declarator cannot be qualified}}

void f6(int A2::RC::x); // expected-error{{parameter declarator cannot be qualified}}

int A2::RC::x; // expected-error{{non-static data member defined out-of-line}}

void A2::CC::NC::m(); // expected-error{{out-of-line declaration of a member must be a definition}}


namespace E {
  int X = 5;
  
  namespace Nested {
    enum E {
      X = 0
    };

    int f() {
      return E::X; // expected-warning{{use of enumeration in a nested name specifier is a C++11 extension}}
    }
  }
}


class Operators {
  Operators operator+(const Operators&) const; // expected-note{{member declaration does not match because it is const qualified}}
  operator bool();
};

Operators Operators::operator+(const Operators&) { // expected-error{{out-of-line definition of 'operator+' does not match any declaration in 'Operators'}}
  Operators ops;
  return ops;
}

Operators Operators::operator+(const Operators&) const {
  Operators ops;
  return ops;
}

Operators::operator bool() {
  return true;
}

namespace A {
  void g(int&); // expected-note{{type of 1st parameter of member declaration does not match definition ('int &' vs 'const int &')}}
} 

void A::f() {} // expected-error-re{{out-of-line definition of 'f' does not match any declaration in namespace 'A'{{$}}}}

void A::g(const int&) { } // expected-error{{out-of-line definition of 'g' does not match any declaration in namespace 'A'}}

struct Struct { };

void Struct::f() { } // expected-error{{out-of-line definition of 'f' does not match any declaration in 'Struct'}}

void global_func(int);
void global_func2(int);

namespace N {
  void ::global_func(int) { } // expected-error{{definition or redeclaration of 'global_func' cannot name the global scope}}

  void f();
  // FIXME: if we move this to a separate definition of N, things break!
}
void ::global_func2(int) { } // expected-warning{{extra qualification on member 'global_func2'}}

void N::f() { } // okay

struct Y;  // expected-note{{forward declaration of 'Y'}}
Y::foo y; // expected-error{{incomplete type 'Y' named in nested name specifier}}

namespace PR25156 {
struct Y;  // expected-note{{forward declaration of 'PR25156::Y'}}
void foo() {
  Y::~Y(); // expected-error{{incomplete type 'PR25156::Y' named in nested name specifier}}
}
}

X::X() : a(5) { } // expected-error{{use of undeclared identifier 'X'}}

struct foo_S {
  static bool value;
};
bool (foo_S::value);


namespace somens {
  struct a { }; // expected-note{{candidate constructor (the implicit copy constructor)}}
}

template <typename T>
class foo {
};


// PR4452 / PR4451
foo<somens:a> a2;  // expected-error {{unexpected ':' in nested name specifier}}

somens::a a3 = a2; // expected-error {{no viable conversion}}

// typedefs and using declarations.
namespace test1 {
  namespace ns {
    class Counter { public: static int count; };
    typedef Counter counter;
  }
  using ns::counter;

  class Test {
    void test1() {
      counter c;
      c.count++;
      counter::count++;
    }
  };
}

// We still need to do lookup in the lexical scope, even if we push a
// non-lexical scope.
namespace test2 {
  namespace ns {
    extern int *count_ptr;
  }
  namespace {
    int count = 0;
  }

  int *ns::count_ptr = &count;
}

// PR6259, invalid case
namespace test3 {
  class A; // expected-note {{forward declaration}}
  void foo(const char *path) {
    A::execute(path); // expected-error {{incomplete type 'test3::A' named in nested name specifier}}
  }
}

namespace PR7133 {
  namespace A {
    class Foo;
  }

  namespace A {
    namespace B {
      bool foo(Foo &);
    }
  }

  bool A::B::foo(Foo &) {
    return false;
  }
}

class CLASS {
  void CLASS::foo2(); // expected-error {{extra qualification on member 'foo2'}}
};

namespace PR8159 {
  class B { };

  class A {
    int A::a; // expected-error{{extra qualification on member 'a'}}
    static int A::b; // expected-error{{extra qualification on member 'b'}}
    int ::c; // expected-error{{non-friend class member 'c' cannot have a qualified name}}
  };
}

namespace rdar7980179 {
  class A { void f0(); }; // expected-note {{previous}}
  int A::f0() {} // expected-error {{return type of out-of-line definition of 'rdar7980179::A::f0' differs}}
}

namespace alias = A;
double *dp = (alias::C*)0; // expected-error{{cannot initialize a variable of type 'double *' with an rvalue of type 'alias::C *'}}

// http://llvm.org/PR10109
namespace PR10109 {
template<typename T>
struct A {
protected:
  struct B;
  struct B::C; // expected-error {{requires a template parameter list}} \
               // expected-error {{no struct named 'C'}} \
    // expected-error{{non-friend class member 'C' cannot have a qualified name}}
};

template<typename T>
struct A2 {
protected:
  struct B;
};
template <typename T>
struct A2<T>::B::C; // expected-error {{no struct named 'C'}}
}

namespace PR13033 {
namespace NS {
 int a; // expected-note {{'NS::a' declared here}}
 int longer_b; //expected-note {{'NS::longer_b' declared here}}
}

// Suggest adding a namespace qualifier to both variable names even though one
// is only a single character long.
int foobar = a + longer_b; // expected-error {{use of undeclared identifier 'a'; did you mean 'NS::a'?}} \
                           // expected-error {{use of undeclared identifier 'longer_b'; did you mean 'NS::longer_b'?}}
}

// <rdar://problem/13853540>
namespace N {
  struct X { };
  namespace N {
    struct Foo {
      struct N::X *foo(); // expected-error{{no struct named 'X' in namespace 'N::N'}}
    };
  }
}

namespace TypedefNamespace { typedef int F; };
TypedefNamespace::F::NonexistentName BadNNSWithCXXScopeSpec; // expected-error {{'TypedefNamespace::F' (aka 'int') is not a class, namespace, or enumeration}}

namespace PR18587 {

struct C1 {
  int a, b, c;
  typedef int C2;
  struct B1 {
    struct B2 {
      int a, b, c;
    };
  };
};
struct C2 { static const unsigned N1 = 1; };
struct B1 {
  enum E1 { B2 = 2 };
  static const int B3 = 3;
};
const int N1 = 2;

// Function declarators
struct S1a { int f(C1::C2); };
struct S1b { int f(C1:C2); };  // expected-error{{unexpected ':' in nested name specifier; did you mean '::'?}}

struct S2a {
  C1::C2 f(C1::C2);
};
struct S2c {
  C1::C2 f(C1:C2);  // expected-error{{unexpected ':' in nested name specifier; did you mean '::'?}}
};

struct S3a {
  int f(C1::C2), C2 : N1;
  int g : B1::B2;
};
struct S3b {
  int g : B1:B2;  // expected-error{{unexpected ':' in nested name specifier; did you mean '::'?}}
};

// Inside square brackets
struct S4a {
  int f[C2::N1];
};
struct S4b {
  int f[C2:N1];  // expected-error{{unexpected ':' in nested name specifier; did you mean '::'?}}
};

struct S5a {
  int f(int xx[B1::B3 ? C2::N1 : B1::B2]);
};
struct S5b {
  int f(int xx[B1::B3 ? C2::N1 : B1:B2]);  // expected-error{{unexpected ':' in nested name specifier; did you mean '::'?}}
};
struct S5c {
  int f(int xx[B1:B3 ? C2::N1 : B1::B2]);  // expected-error{{unexpected ':' in nested name specifier; did you mean '::'?}}
};

// Bit fields
struct S6a {
  C1::C2 m1 : B1::B2;
};
struct S6c {
  C1::C2 m1 : B1:B2;  // expected-error{{unexpected ':' in nested name specifier; did you mean '::'?}}
};
struct S6d {
  int C2:N1;
};
struct S6e {
  static const int N = 3;
  B1::E1 : N;
};
struct S6g {
  C1::C2 : B1:B2;  // expected-error{{unexpected ':' in nested name specifier; did you mean '::'?}}
  B1::E1 : B1:B2;  // expected-error{{unexpected ':' in nested name specifier; did you mean '::'?}}
};

// Template parameters
template <int N> struct T1 {
  int a,b,c;
  static const unsigned N1 = N;
  typedef unsigned C1;
};
T1<C2::N1> var_1a;
T1<C2:N1> var_1b;  // expected-error{{unexpected ':' in nested name specifier; did you mean '::'?}}
template<int N> int F() {}
int (*X1)() = (B1::B2 ? F<1> : F<2>);
int (*X2)() = (B1:B2 ? F<1> : F<2>);  // expected-error{{unexpected ':' in nested name specifier; did you mean '::'?}}

// Bit fields + templates
struct S7a {
  T1<B1::B2>::C1 m1 : T1<B1::B2>::N1;
};
struct S7b {
  T1<B1:B2>::C1 m1 : T1<B1::B2>::N1;  // expected-error{{unexpected ':' in nested name specifier; did you mean '::'?}}
};
struct S7c {
  T1<B1::B2>::C1 m1 : T1<B1:B2>::N1;  // expected-error{{unexpected ':' in nested name specifier; did you mean '::'?}}
};

}

namespace PR16951 {
  namespace ns {
    enum an_enumeration {
      ENUMERATOR  // expected-note{{'ENUMERATOR' declared here}}
    };
  }

  int x1 = ns::an_enumeration::ENUMERATOR; // expected-warning{{use of enumeration in a nested name specifier is a C++11 extension}}

  int x2 = ns::an_enumeration::ENUMERATOR::vvv; // expected-warning{{use of enumeration in a nested name specifier is a C++11 extension}} \
                                                // expected-error{{'ENUMERATOR' is not a class, namespace, or enumeration}} \

  int x3 = ns::an_enumeration::X; // expected-warning{{use of enumeration in a nested name specifier is a C++11 extension}} \
                                  // expected-error{{no member named 'X'}}

  enum enumerator_2 {
    ENUMERATOR_2
  };

  int x4 = enumerator_2::ENUMERATOR_2; // expected-warning{{use of enumeration in a nested name specifier is a C++11 extension}}
  int x5 = enumerator_2::X2; // expected-warning{{use of enumeration in a nested name specifier is a C++11 extension}} \
                             // expected-error{{no member named 'X2' in 'PR16951::enumerator_2'}}

}

namespace PR30619 {
c d; c d; c d; c d; c d; c d; c d; c d; c d; c d; c d; c d; c d; c d; c d; c d;
// expected-error@-1 16{{unknown type name 'c'}}
c d; c d; c d; c d; c d; c d; c d; c d; c d; c d; c d; c d; c d; c d; c d; c d;
// expected-error@-1 16{{unknown type name 'c'}}
c d; c d; c d; c d; c d; c d; c d; c d; c d; c d; c d; c d; c d; c d; c d; c d;
// expected-error@-1 16{{unknown type name 'c'}}
c d; c d; c d; c d; c d; c d; c d; c d; c d; c d; c d; c d; c d; c d; c d; c d;
// expected-error@-1 16{{unknown type name 'c'}}
namespace A {
class B {
  typedef C D; // expected-error{{unknown type name 'C'}}
  A::D::F;
  // expected-error@-1{{'PR30619::A::B::D' (aka 'int') is not a class, namespace, or enumeration}}
};
}
}

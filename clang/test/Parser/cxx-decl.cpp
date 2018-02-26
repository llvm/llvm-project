// RUN: %clang_cc1 -verify -fsyntax-only -triple i386-linux -Wredundant-parens -pedantic-errors -fcxx-exceptions -fexceptions %s
// RUN: %clang_cc1 -verify -fsyntax-only -triple i386-linux -Wredundant-parens -pedantic-errors -fcxx-exceptions -fexceptions -std=c++98 %s
// RUN: %clang_cc1 -verify -fsyntax-only -triple i386-linux -Wredundant-parens -pedantic-errors -fcxx-exceptions -fexceptions -std=c++11 %s

const char const *x10; // expected-error {{duplicate 'const' declaration specifier}}

int x(*g); // expected-error {{use of undeclared identifier 'g'}}

struct Type {
  int Type;
};

// rdar://8365458
// rdar://9132143
typedef char bool; // expected-error {{redeclaration of C++ built-in type 'bool'}}

// PR4451 - We should recover well from the typo of '::' as ':' in a2.
namespace y {
  struct a { };
  typedef int b;
}

y::a a1;
y:a a2;  // expected-error {{unexpected ':' in nested name specifier}}
y::a a3 = a2;

// Some valid colons:
void foo() {
y:  // label
  y::a s;
  
  int a = 4;
  a = a ? a : a+1;
}

struct b : y::a {};

template <typename T>
class someclass {
  
  int bar() {
    T *P;
    return 1 ? P->x : P->y;
  }
};

class asm_class_test {
  void foo() __asm__("baz");
};

enum { fooenum = 1, };
#if __cplusplus <= 199711L
// expected-error@-2 {{commas at the end of enumerator lists are a C++11 extension}}
#endif

struct a {
  int Type : fooenum;
};

void test(struct Type *P) {
  int Type;
  Type = 1 ? P->Type : Type;
  
  Type = (y:b) 4;   // expected-error {{unexpected ':' in nested name specifier}}
  Type = 1 ? (
              (y:b)  // expected-error {{unexpected ':' in nested name specifier}}
              4) : 5;
}

struct test4 {
  int x  // expected-error {{expected ';' at end of declaration list}}
  int y;
  int z  // expected-error {{expected ';' at end of declaration list}}
};

// Make sure we know these are legitimate commas and not typos for ';'.
namespace Commas {
  struct S {
    static int a;
    int c,
    operator()();
  };

  int global1,
  __attribute__(()) global2,
  (global5), // expected-warning {{redundant parentheses surrounding declarator}}
  *global6,
  &global7 = global1,
  &&global8 = static_cast<int&&>(global1),
#if __cplusplus <= 199711L
  // expected-error@-2 2{{rvalue references are a C++11 extension}}
#endif

  S::a,
  global9,
  global10 = 0,
  global11 == 0, // expected-error {{did you mean '='}}
  global12 __attribute__(()),
  global13(0),
  global14[2],
  global15;

  void g() {
    static int a,
    b __asm__("ebx"), // expected-error {{expected ';' at end of declaration}}
    Statics:return;
  }
}

// PR5825
struct test5 {};
::new(static_cast<void*>(0)) test5; // expected-error {{expected unqualified-id}}


// PR6782
template<class T>
class Class1;

class Class2 {
} // expected-error {{expected ';' after class}}

typedef Class1<Class2> Type1;

// rdar : // 8307865
struct CodeCompleteConsumer {
};

void CodeCompleteConsumer::() { // expected-error {{xpected unqualified-id}}
} 

;

// PR4111
void f(sqrgl); // expected-error {{unknown type name 'sqrgl'}}

// PR9903
struct S {
  typedef void a() { }; // expected-error {{function definition declared 'typedef'}}
  typedef void c() try { } catch(...) { } // expected-error {{function definition declared 'typedef'}}
  int n, m;
  typedef S() : n(1), m(2) { } // expected-error {{function definition declared 'typedef'}}
};


namespace TestIsValidAfterTypeSpecifier {
struct s {} v;

namespace a {
struct s operator++(struct s a)
{ return a; }
}

namespace b {
// The newline after s should make no difference.
struct s
operator++(struct s a)
{ return a; }
}

struct X {
  struct s
  friend f();
  struct s
  virtual f();
};

struct s
&r0 = v;
struct s
bitand r2 = v;

}

struct DIE {
  void foo() {}
};

void test (DIE die, DIE *Die, DIE INT, DIE *FLOAT) {
  DIE.foo();  // expected-error {{cannot use dot operator on a type}}
  die.foo();

  DIE->foo();  // expected-error {{cannot use arrow operator on a type}}
  Die->foo();

  int.foo();  // expected-error {{cannot use dot operator on a type}}
  INT.foo();

  float->foo();  // expected-error {{cannot use arrow operator on a type}}
  FLOAT->foo();
}

namespace PR15017 {
  template<typename T = struct X { int i; }> struct S {}; // expected-error {{'PR15017::X' cannot be defined in a type specifier}}
}

// Ensure we produce at least some diagnostic for attributes in C++98.
[[]] struct S;
#if __cplusplus <= 199711L
// expected-error@-2 {{expected expression}}
// expected-error@-3 {{expected unqualified-id}}
#else
// expected-error@-5 {{misplaced attributes}}
#endif

namespace test7 {
  struct Foo {
    void a();
    void b();
  };

  void Foo::
  // Comment!
  a() {}


  void Foo::  // expected-error {{expected unqualified-id}}
  // Comment!
}

void test8() {
  struct {} o;
  // This used to crash.
  (&o)->(); // expected-error{{expected unqualified-id}}
}

namespace PR5066 {
  template<typename T> struct X {};
  X<int N> x; // expected-error {{type-id cannot have a name}}

  using T = int (*T)(); // expected-error {{type-id cannot have a name}}
#if __cplusplus <= 199711L
  // expected-error@-2 {{alias declarations are a C++11 extensio}}
#endif

}

namespace PR17255 {
void foo() {
  typename A::template B<>; // expected-error {{use of undeclared identifier 'A'}}
#if __cplusplus <= 199711L
  // expected-error@-2 {{'template' keyword outside of a template}}
#endif
  // expected-error@-4 {{expected a qualified name after 'typename'}}
}
}

namespace PR17567 {
  struct Foobar { // expected-note 2{{declared here}}
    FooBar(); // expected-error {{missing return type for function 'FooBar'; did you mean the constructor name 'Foobar'?}}
    ~FooBar(); // expected-error {{expected the class name after '~' to name a destructor}}
  };
  FooBar::FooBar() {} // expected-error {{undeclared}} expected-error {{missing return type}}
  FooBar::~FooBar() {} // expected-error {{undeclared}} expected-error {{expected the class name}}
}

namespace DuplicateFriend {
  struct A {
    friend void friend f(); // expected-warning {{duplicate 'friend' declaration specifier}}
    friend struct B friend; // expected-warning {{duplicate 'friend' declaration specifier}}
#if __cplusplus >= 201103L
    // expected-error@-2 {{'friend' must appear first in a non-function declaration}}
#endif
  };
}

namespace NNS {
  struct A {};
  namespace B { extern A C1, C2, *C3, C4[], C5; }
  // Do not produce a redundant parentheses warning here; removing these parens
  // changes the meaning of the program.
  A (::NNS::B::C1);
  A (NNS::B::C2); // expected-warning {{redundant parentheses surrounding declarator}}
  A (*::NNS::B::C3); // expected-warning {{redundant parentheses surrounding declarator}}
  A (::NNS::B::C4[2]);
  // Removing one of these sets of parentheses would be reasonable.
  A ((::NNS::B::C5)); // expected-warning {{redundant parentheses surrounding declarator}}

  void f() {
    // FIXME: A vexing-parse warning here would be useful.
    A(::NNS::B::C1); // expected-error {{definition or redeclaration}}
    A(NNS::B::C1); // expected-warning {{redundant paren}} expected-error {{definition or redeclaration}}
  }
}

inline namespace ParensAroundFriend { // expected-error 0-1{{C++11}}
  struct A {};
  struct B {
    static A C();
  };
  namespace X {
    struct B {};
    struct D {
      // No warning here: while this could be written as
      //   friend (::B::C)();
      // we do need parentheses *somewhere* here.
      friend A (::B::C());
    };
  }
}

// PR8380
extern ""      // expected-error {{unknown linkage language}}
test6a { ;// expected-error {{C++ requires a type specifier for all declarations}}
#if __cplusplus <= 199711L
// expected-error@-2 {{expected ';' after top level declarator}}
#else
// expected-error@-4 {{expected expression}}
// expected-note@-5 {{to match this}}
#endif
  
  int test6b;
#if __cplusplus >= 201103L
// expected-error@+3 {{expected}}
// expected-error@-3 {{expected ';' after top level declarator}}
#endif


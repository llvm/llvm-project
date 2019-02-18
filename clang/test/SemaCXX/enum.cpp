// RUN: %clang_cc1 -fsyntax-only -pedantic -std=c++98 -verify -triple x86_64-apple-darwin %s
// RUN: %clang_cc1 -fsyntax-only -pedantic -std=c++11 -verify -triple x86_64-apple-darwin %s
enum E { // expected-note{{previous definition is here}}
  Val1,
  Val2
};

enum E; // expected-warning{{redeclaration of already-defined enum 'E' is a GNU extension}}

int& enumerator_type(int);
float& enumerator_type(E);

void f() {
  E e = Val1;
  float& fr = enumerator_type(Val2);
}

// <rdar://problem/6502934>
typedef enum Foo {
  A = 0,
  B = 1
} Foo;

void bar() {
  Foo myvar = A;
  myvar = B;
}

/// PR3688
struct s1 {
  enum e1 (*bar)(void); // expected-error{{ISO C++ forbids forward references to 'enum' types}}
};

enum e1 { YES, NO };

static enum e1 badfunc(struct s1 *q) {
  return q->bar();
}

enum e2; // expected-error{{ISO C++ forbids forward references to 'enum' types}}

namespace test1 {
  template <class A, class B> struct is_same { static const int value = -1; };
  template <class A> struct is_same<A,A> { static const int value = 1; };

  enum enum0 { v0 };
  int test0[is_same<__typeof(+v0), int>::value];

  enum enum1 { v1 = __INT_MAX__ };
  int test1[is_same<__typeof(+v1), int>::value];

  enum enum2 { v2 = __INT_MAX__ * 2U };
  int test2[is_same<__typeof(+v2), unsigned int>::value];

  enum enum3 { v3 = __LONG_MAX__ };
  int test3[is_same<__typeof(+v3), long>::value];

  enum enum4 { v4 = __LONG_MAX__ * 2UL };
  int test4[is_same<__typeof(+v4), unsigned long>::value];
}

// PR6061
namespace PR6061 {
  struct A { enum { id }; };
  struct B { enum { id }; };
  
  struct C : public A, public B
  { 
    enum { id }; 
  };
}

namespace Conditional {
  enum a { A }; a x(const enum a x) { return 1?x:A; }
}

namespace PR7051 {
  enum E { e0 };
  void f() {
    E e;
    e = 1; // expected-error{{assigning to 'PR7051::E' from incompatible type 'int'}}
    e |= 1; // expected-error{{assigning to 'PR7051::E' from incompatible type 'int'}}
  }
}

// PR7466
enum { }; // expected-warning{{declaration does not declare anything}}
typedef enum { }; // expected-warning{{typedef requires a name}}

// PR7921
enum PR7921E {
    PR7921V = (PR7921E)(123)
#if __cplusplus < 201103L
// expected-error@-2 {{expression is not an integral constant expression}}
#else
// expected-error@-4 {{must have integral or unscoped enumeration type}}
// FIXME: The above diagnostic isn't very good; we should instead complain about the type being incomplete.
#endif
};

void PR8089() {
  enum E; // expected-error{{ISO C++ forbids forward references to 'enum' types}}
  int a = (E)3; // expected-error{{cannot initialize a variable of type 'int' with an rvalue of type 'E'}}
}

// This is accepted as a GNU extension. In C++98, there was no provision for
// expressions with UB to be non-constant.
enum { overflow = 123456 * 234567 };
#if __cplusplus >= 201103L
// expected-warning@-2 {{not an integral constant expression}}
// expected-note@-3 {{value 28958703552 is outside the range of representable values}}
#else 
// expected-warning@-5 {{overflow in expression; result is -1106067520 with type 'int'}}
#endif

// PR28903
struct PR28903 {
  enum {
    PR28903_A = (enum { // expected-error-re {{'PR28903::(anonymous enum at {{.*}})' cannot be defined in an enumeration}}
      PR28903_B,
      PR28903_C = PR28903_B
    })
  };
};

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
    e = 1; // expected-error{{assigning to 'E' from incompatible type 'int'}}
    e |= 1; // expected-error{{assigning to 'E' from incompatible type 'int'}}
  }
}

// PR7466
enum { }; // expected-warning{{declaration does not declare anything}}
typedef enum { }; // expected-warning{{typedef requires a name}}

// PR7921
enum PR7921E { // expected-note {{not complete until the closing '}'}}
    PR7921V = (PR7921E)(123) // expected-error {{'PR7921E' is an incomplete type}}
};

void PR8089() {
  enum E; // expected-error{{ISO C++ forbids forward references to 'enum' types}} expected-note {{forward declaration}}
  int a = (E)3; // expected-error {{'E' is an incomplete type}}
}

// This is accepted as a GNU extension. In C++98, there was no provision for
// expressions with UB to be non-constant.
enum { overflow = 123456 * 234567 };
#if __cplusplus >= 201103L
// expected-warning@-2 {{not an integral constant expression}}
// expected-note@-3 {{value 28958703552 is outside the range of representable values}}
#else 
// expected-warning@-5 {{overflow in expression; result is -1'106'067'520 with type 'int'}}
#endif

// FIXME: This is not consistent with the above case.
enum NoFold : int { overflow2 = 123456 * 234567 };
#if __cplusplus >= 201103L
// expected-error@-2 {{enumerator value is not a constant expression}}
// expected-note@-3 {{value 28958703552 is outside the range of representable values}}
#else
// expected-warning@-5 {{overflow in expression; result is -1'106'067'520 with type 'int'}}
// expected-warning@-6 {{extension}}
#endif

// PR28903
struct PR28903 {
  enum {
    PR28903_A = (enum { // expected-error-re {{'PR28903::(unnamed enum at {{.*}})' cannot be defined in an enumeration}}
      PR28903_B,
      PR28903_C = PR28903_B
    })
  };
};

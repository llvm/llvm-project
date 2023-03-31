// RUN: %clang_cc1 -verify -std=c++20 %s -fsyntax-only
// RUN: %clang_cc1 -verify=expected,beforecxx20 -Wc++20-extensions -std=c++20 %s -fsyntax-only

struct A { // expected-note 4{{candidate constructor}}
  char i;
  double j;
};

struct B {
  A a;
  int b[20];
  int &&c; // expected-note {{reference member declared here}}
};

struct C { // expected-note 5{{candidate constructor}}
  A a;
  int b[20];
};

struct D : public C, public A {
  int a;
};

struct E { // expected-note 3{{candidate constructor}}
  struct F {
    F(int, int);
  };
  int a;
  F f;
};

int getint(); // expected-note {{declared here}}

struct F {
  int a;
  int b = getint(); // expected-note {{non-constexpr function 'getint' cannot be used in a constant expression}}
};

template <typename T>
struct G {
  T t1;
  T t2;
};

struct H {
  virtual void foo() = 0;
};

struct I : public H { // expected-note 3{{candidate constructor}}
  int i, j;
  void foo() override {}
};

struct J {
  int a;
  int b[]; // expected-note {{initialized flexible array member 'b' is here}}
};

union U {
  int a;
  char* b;
};

template <typename T, char CH>
void bar() {
  T t = 0;
  A a(CH, 1.1); // OK; C++ paren list constructors are supported in semantic tree transformations.
  // beforecxx20-warning@-1 2{{aggregate initialization of type 'A' from a parenthesized list of values is a C++20 extension}}
}

template <class T, class... Args>
T Construct(Args... args) {
  return T(args...); // OK; variadic arguments can be used in paren list initializers.
  // beforecxx20-warning@-1 {{aggregate initialization of type 'A' from a parenthesized list of values is a C++20 extension}}
}

void foo() {
  A a1(1954, 9, 21);
  // expected-error@-1 {{excess elements in struct initializer}}
  A a2(2.1);
  // expected-warning@-1 {{implicit conversion from 'double' to 'char'}}
  // beforecxx20-warning@-2 {{aggregate initialization of type 'A' from a parenthesized list of values is a C++20 extension}}
  A a3(-1.2, 9.8);
  // expected-warning@-1 {{implicit conversion from 'double' to 'char'}}
  // beforecxx20-warning@-2 {{aggregate initialization of type 'A' from a parenthesized list of values is a C++20 extension}}
  A a4 = static_cast<A>(1.1);
  // expected-warning@-1 {{implicit conversion from 'double' to 'char'}}
  // beforecxx20-warning@-2 {{aggregate initialization of type 'A' from a parenthesized list of values is a C++20 extension}}
  A a5 = (A)3.1;
  // expected-warning@-1 {{implicit conversion from 'double' to 'char'}}
  // beforecxx20-warning@-2 {{aggregate initialization of type 'A' from a parenthesized list of values is a C++20 extension}}
  A a6 = A(8.7);
  // expected-warning@-1 {{implicit conversion from 'double' to 'char'}}
  // beforecxx20-warning@-2 {{aggregate initialization of type 'A' from a parenthesized list of values is a C++20 extension}}

  B b1(2022, {7, 8});
  // expected-error@-1 {{no viable conversion from 'int' to 'A'}}
  B b2(A(1), {}, 1);
  // expected-error@-1 {{reference member 'c' binds to a temporary object whose lifetime would be shorter than the lifetime of the constructed object}}
  // beforecxx20-warning@-2 {{aggregate initialization of type 'A' from a parenthesized list of values is a C++20 extension}}
  // beforecxx20-warning@-3 {{aggregate initialization of type 'B' from a parenthesized list of values is a C++20 extension}}

  C c(A(1), 1, 2, 3, 4);
  // expected-error@-1 {{array initializer must be an initializer list}}
  // beforecxx20-warning@-2 {{aggregate initialization of type 'A' from a parenthesized list of values is a C++20 extension}}
  D d1(1);
  // expected-error@-1 {{no viable conversion from 'int' to 'C'}}
  D d2(C(1));
  // expected-error@-1 {{no matching conversion for functional-style cast from 'int' to 'C'}}
  // beforecxx20-warning@-2 {{aggregate initialization of type 'D' from a parenthesized list of values is a C++20 extension}}
  D d3(C(A(1)), 1);
  // expected-error@-1 {{no viable conversion from 'int' to 'A'}}
  // beforecxx20-warning@-2 {{aggregate initialization of type 'A' from a parenthesized list of values is a C++20 extension}}
  // beforecxx20-warning@-3 {{aggregate initialization of type 'C' from a parenthesized list of values is a C++20 extension}}

  int arr1[](0, 1, 2, A(1));
  // expected-error@-1 {{no viable conversion from 'A' to 'int'}}
  // beforecxx20-warning@-2 {{aggregate initialization of type 'A' from a parenthesized list of values is a C++20 extension}}

  int arr2[2](0, 1, 2);
  // expected-error@-1 {{excess elements in array initializer}}

  // We should not build paren list initilizations for IK_COPY.
  int arr3[1] = 1;
  // expected-error@-1 {{array initializer must be an initializer list}}

  U u1("abcd");
  // expected-error@-1 {{cannot initialize a member subobject of type 'int' with an lvalue of type 'const char[5]'}}
  U u2(1, "efgh");
  // expected-error@-1 {{excess elements in union initializer}}

  E e1(1);
  // expected-error@-1 {{no matching constructor for initialization of 'E'}}

  constexpr F f1(1);
  // expected-error@-1 {{constexpr variable 'f1' must be initialized by a constant expression}}
  // beforecxx20-warning@-2 {{aggregate initialization of type 'const F' from a parenthesized list of values is a C++20 extension}}

  constexpr F f2(1, 1); // OK: f2.b is initialized by a constant expression.
  // beforecxx20-warning@-1 {{aggregate initialization of type 'const F' from a parenthesized list of values is a C++20 extension}}

  bar<int, 'a'>();
  // beforecxx20-note@-1 {{in instantiation of function template specialization 'bar<int, 'a'>' requested here}}

  G<char> g('b', 'b');
  // beforecxx20-warning@-1 {{aggregate initialization of type 'G<char>' from a parenthesized list of values is a C++20 extension}}

  A a7 = Construct<A>('i', 2.2);
  // beforecxx20-note@-1 {{in instantiation of function template specialization 'Construct<A, char, double>' requested here}}

  int arr4[](1, 2);
  // beforecxx20-warning@-1 {{aggregate initialization of type 'int[2]' from a parenthesized list of values is a C++20 extension}}

  int arr5[2](1, 2);
  // beforecxx20-warning@-1 {{aggregate initialization of type 'int[2]' from a parenthesized list of values is a C++20 extension}}

  I i(1, 2);
  // expected-error@-1 {{no matching constructor for initialization of 'I'}}

  J j(1, {2, 3});
  // expected-error@-1 {{initialization of flexible array member is not allowed}}

  static_assert(__is_trivially_constructible(A, char, double));
  static_assert(__is_trivially_constructible(A, char, int));
  static_assert(__is_trivially_constructible(A, char));

  static_assert(__is_trivially_constructible(D, C, A, int));
  static_assert(__is_trivially_constructible(D, C));

  static_assert(__is_trivially_constructible(int[2], int, int));
  static_assert(__is_trivially_constructible(int[2], int, double));
  static_assert(__is_trivially_constructible(int[2], int));
}

namespace gh59675 {
struct K {
  template <typename T>
  K(T);

  virtual ~K();
};

union V {
  K k;
  // expected-note@-1 {{default constructor of 'V' is implicitly deleted because field 'k' has no default constructor}}
  // expected-note@-2 2{{copy constructor of 'V' is implicitly deleted because variant field 'k' has a non-trivial copy constructor}}
};

static_assert(!__is_constructible(V, const V&));
static_assert(!__is_constructible(V, V&&));

void bar() {
  V v1;
  // expected-error@-1 {{call to implicitly-deleted default constructor of 'V'}}

  V v2(v1);
  // expected-error@-1 {{call to implicitly-deleted copy constructor of 'V'}}

  V v3((V&&) v1);
  // expected-error@-1 {{call to implicitly-deleted copy constructor of 'V'}}
}
}

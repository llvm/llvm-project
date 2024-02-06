// RUN: %clang_cc1 -verify -std=c++20 %s -fsyntax-only
// RUN: %clang_cc1 -verify -std=c++20 %s -fsyntax-only -fexperimental-new-constant-interpreter
// RUN: %clang_cc1 -verify=expected,beforecxx20 -Wc++20-extensions -std=c++20 %s -fsyntax-only -fexperimental-new-constant-interpreter
// RUN: %clang_cc1 -verify=expected,beforecxx20 -Wc++20-extensions -std=c++20 %s -fsyntax-only

struct A { // expected-note 4{{candidate constructor}}
  char i;
  double j;
};

struct B {
  A a;
  int b[20];
  int &&c;
};

struct C { // expected-note 5{{candidate constructor}}
  A a;
  int b[20];
};

struct D : public C, public A {
  int a;
};

struct E {
  struct F { // expected-note 2{{candidate constructor}}
    F(int, int); // expected-note {{candidate constructor}}
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

enum K { K0, K1, K2 };

struct L {
  K k : 1;
};

struct M {
  struct N {
    private:
    N(int);
    // expected-note@-1 {{declared private here}}
  };
  int i;
  N n;
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

void foo(int n) { // expected-note {{declared here}}
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
  // beforecxx20-warning@-1 {{aggregate initialization of type 'A' from a parenthesized list of values is a C++20 extension}}
  // beforecxx20-warning@-2 {{aggregate initialization of type 'B' from a parenthesized list of values is a C++20 extension}}

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
  // expected-error@-1 {{no matching constructor for initialization of 'F'}}

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

  L l(K::K2);
  // expected-warning@-1 {{implicit truncation}}
  // beforecxx20-warning@-2 {{aggregate initialization of type 'L' from a parenthesized list of values is a C++20 extension}}

  int arr4[](1, 2);
  // beforecxx20-warning@-1 {{aggregate initialization of type 'int[2]' from a parenthesized list of values is a C++20 extension}}

  int arr5[2](1, 2);
  // beforecxx20-warning@-1 {{aggregate initialization of type 'int[2]' from a parenthesized list of values is a C++20 extension}}

  int arr6[n](1, 2, 3); // expected-warning {{variable length arrays in C++ are a Clang extension}} \
                           expected-note {{function parameter 'n' with unknown value cannot be used in a constant expression}} \
                           expected-error {{variable-sized object may not be initialized}}

  I i(1, 2);
  // expected-error@-1 {{no matching constructor for initialization of 'I'}}

  J j(1, {2, 3});
  // expected-error@-1 {{initialization of flexible array member is not allowed}}

  M m(1, 1);
  // expected-error@-1 {{field of type 'N' has private constructor}}
  // beforecxx20-warning@-2 {{aggregate initialization of type 'M' from a parenthesized list of values is a C++20 extension}}

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

namespace gh62296 {
struct L {
protected:
  L(int);
  // expected-note@-1 2{{declared protected here}}
};

struct M : L {};

struct N {
  L l;
};

M m(42);
// expected-error@-1 {{base class 'L' has protected constructor}}
// beforecxx20-warning@-2 {{aggregate initialization of type 'M' from a parenthesized list of values is a C++20 extension}}

N n(43);
// expected-error@-1 {{field of type 'L' has protected constructor}}
// beforecxx20-warning@-2 {{aggregate initialization of type 'N' from a parenthesized list of values is a C++20 extension}}
}

namespace gh61567 {
struct O {
  int i;
  int &&j;
  // expected-note@-1 {{uninitialized reference member is here}}
  int &&k = 1;
};

O o1(0, 0, 0); // no-error
// beforecxx20-warning@-1 {{aggregate initialization of type 'O' from a parenthesized list of values is a C++20 extension}}

O o2(0, 0); // no-error
// beforecxx20-warning@-1 {{aggregate initialization of type 'O' from a parenthesized list of values is a C++20 extension}}

O o3(0);
// expected-error@-1 {{reference member of type 'int &&' uninitialized}}
}

namespace gh63008 {
auto a = new A('a', {1.1});
// expected-warning@-1 {{braces around scalar init}}
// beforecxx20-warning@-2 {{aggregate initialization of type 'A' from a parenthesized list of values is a C++20 extension}}
}


namespace GH63278 {
struct S {
  int a = 0;
  int b {0};
  auto x = 1; // expected-error {{'auto' not allowed in non-static struct member}}
  static const auto y = 1;
};

int test() {
  // used to crash
  S a(0, 1);
  S b(0);
  S c(0, 0, 1);

  S d {0, 1};
  S e {0};
  S f {0, 0, 1};
}

}

namespace gh63758 {
  struct S {} s;
  auto words = (char[])s; // expected-error {{C-style cast from 'struct S' to 'char[]' is not allowed}}
};

namespace GH63903 {
  constexpr int f(); // expected-note {{declared here}}
  struct S {
    int a = 0, b = f(); // expected-note {{undefined function 'f' cannot be used in a constant expression}}
  };

  // Test that errors produced by default members are produced at the location of the initialization
  constexpr S s(0); // beforecxx20-warning {{aggregate initialization of type 'const S' from a parenthesized list of values is a C++20 extension}} \
                    // expected-error {{constexpr variable 's' must be initialized by a constant expression}}
}

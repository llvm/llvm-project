// RUN: %clang_cc1 -std=c2x -fsyntax-only -verify -Wno-unused %s
// RUN: %clang_cc1 -fsyntax-only -verify=expected,cpp -Wno-unused -x c++ -std=c++17 %s

// Test various parsing situations for the Clang extension to _Generic which
// accepts a type name instead of an expression as the first operand.

int foo();

void test() {
  // We can parse a simple type name.
  _Generic(int, int : 0);

  // We can also parse tag types.
  struct S { int i; };
  enum E { A };
  union U { int i; };
  _Generic(struct S, default : 0);
  _Generic(enum E, default : 0);
  _Generic(union U, default : 0);
  
  // We can also parse array types.
  _Generic(int[12], default : 0);
  
  // And pointer to array types, too.
  _Generic(int(*)[12], default : 0);
  
  // We do not accept a parenthesized type name.
  _Generic((int), int : 0); // expected-error {{expected expression}}
  
  // We can parse more complex types as well. Note, this is a valid spelling of
  // a function  pointer type in C but is not a valid spelling of a function
  // pointer type in C++. Surprise!
  _Generic(__typeof__(foo())(*)(__typeof__(&foo)), int (*)(int (*)()) : 0); // cpp-error {{expected expression}} \
                                                                               cpp-error {{expected '(' for function-style cast or type construction}}

  // C being the magical language that it is, lets you define a type anywhere
  // you can spell a type.
  _Generic(struct T { int a; }, default : 0); // cpp-error {{'T' cannot be defined in a type specifier}}
}

#ifdef __cplusplus
template <typename Ty>
struct S {
  template <template <typename> typename Uy>
  struct T {
    typedef typename Uy<Ty>::type foo;
  };
};

template <typename Ty>
struct inst {
  typedef Ty type;
};

void cpp_test() {
  // Ensure we can parse more complex C++ typenames as well.
  _Generic(S<int>::T<inst>::foo, int : 1);
  
  // And that the type name doesn't confuse us when given an initialization
  // expression.
  _Generic(S<int>::T<inst>::foo{}, int : 1);
}

template <typename Ty, int N = _Generic(Ty, int : 0, default : 1)>
constexpr Ty bar() { return N; }

static_assert(bar<int>() == 0);
static_assert(bar<float>() == 1);
#endif // __cplusplus

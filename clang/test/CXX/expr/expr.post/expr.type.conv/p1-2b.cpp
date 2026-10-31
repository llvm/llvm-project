// RUN: %clang_cc1 -std=c++20 -verify=expected,cxx20 %s
// RUN: %clang_cc1 -std=c++23 -verify %s

template <class T>
void foo(T);

struct A {
  int m;
  char g(int);
  float g(double);
} a{1};

// C++23 [dcl.type.auto.deduct]p2.3
// For an explicit type conversion, T is the specified type, which shall be auto.
void diagnostics() {
  foo(auto());   // expected-error {{initializer for functional-style cast to 'auto' is empty}} cxx20-warning {{'auto' as a functional-style cast is a C++23 extension}}
  foo(auto{});   // expected-error {{initializer for functional-style cast to 'auto' is empty}} cxx20-warning {{'auto' as a functional-style cast is a C++23 extension}}
  foo(auto({})); // expected-error {{cannot deduce actual type for 'auto' from parenthesized initializer list}} cxx20-warning {{'auto' as a functional-style cast is a C++23 extension}}
  foo(auto{{}}); // expected-error {{cannot deduce actual type for 'auto' from nested initializer list}} cxx20-warning {{'auto' as a functional-style cast is a C++23 extension}}

  // - If the initializer is a parenthesized expression-list, the expression-list shall be a single assignmentexpression and E is the assignment-expression.
  foo(auto(a)); // cxx20-warning {{'auto' as a functional-style cast is a C++23 extension}}
  // - If the initializer is a braced-init-list, it shall consist of a single brace-enclosed assignment-expression and E is the assignment-expression.
  foo(auto{a});   // cxx20-warning {{'auto' as a functional-style cast is a C++23 extension}}
  foo(auto({a})); // expected-error {{cannot deduce actual type for 'auto' from parenthesized initializer list}} cxx20-warning {{'auto' as a functional-style cast is a C++23 extension}}
  foo(auto{{a}}); // expected-error {{cannot deduce actual type for 'auto' from nested initializer list}} cxx20-warning {{'auto' as a functional-style cast is a C++23 extension}}

  foo(auto(&A::g)); // expected-error {{functional-style cast to 'auto' has incompatible initializer of type '<overloaded function type>'}} cxx20-warning {{'auto' as a functional-style cast is a C++23 extension}}

  foo(auto(a, 3.14));     // expected-error {{initializer for functional-style cast to 'auto' contains multiple expressions}} cxx20-warning {{'auto' as a functional-style cast is a C++23 extension}}
  foo(auto{a, 3.14});     // expected-error {{initializer for functional-style cast to 'auto' contains multiple expressions}} cxx20-warning {{'auto' as a functional-style cast is a C++23 extension}}
  foo(auto({a, 3.14}));   // expected-error {{cannot deduce actual type for 'auto' from parenthesized initializer list}} cxx20-warning {{'auto' as a functional-style cast is a C++23 extension}}
  foo(auto{{a, 3.14}});   // expected-error {{cannot deduce actual type for 'auto' from nested initializer list}} cxx20-warning {{'auto' as a functional-style cast is a C++23 extension}}
  foo(auto({a}, {3.14})); // expected-error {{initializer for functional-style cast to 'auto' contains multiple expressions}} cxx20-warning {{'auto' as a functional-style cast is a C++23 extension}}
  foo(auto{{a}, {3.14}}); // expected-error {{initializer for functional-style cast to 'auto' contains multiple expressions}} cxx20-warning {{'auto' as a functional-style cast is a C++23 extension}}

  foo(auto{1, 2});   // expected-error {{initializer for functional-style cast to 'auto' contains multiple expressions}} cxx20-warning {{'auto' as a functional-style cast is a C++23 extension}}
  foo(auto({1, 2})); // expected-error {{cannot deduce actual type for 'auto' from parenthesized initializer list}} cxx20-warning {{'auto' as a functional-style cast is a C++23 extension}}
  foo(auto{{1, 2}}); // expected-error {{cannot deduce actual type for 'auto' from nested initializer list}} cxx20-warning {{'auto' as a functional-style cast is a C++23 extension}}s
}

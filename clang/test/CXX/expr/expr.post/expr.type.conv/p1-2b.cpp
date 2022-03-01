// RUN: %clang_cc1 -std=c++2b -verify %s

template <class T>
void foo(T);

struct A {
  int m;
  char g(int);
  float g(double);
} a{1};

// C++2b [dcl.type.auto.deduct]p2.3
// For an explicit type conversion, T is the specified type, which shall be auto.
void diagnostics() {
  foo(auto());   // expected-error {{initializer for functional-style cast to 'auto' is empty}}
  foo(auto{});   // expected-error {{initializer for functional-style cast to 'auto' is empty}}
  foo(auto({})); // expected-error {{cannot deduce actual type for 'auto' from parenthesized initializer list}}
  foo(auto{{}}); // expected-error {{cannot deduce actual type for 'auto' from nested initializer list}}

  // - If the initializer is a parenthesized expression-list, the expression-list shall be a single assignmentexpression and E is the assignment-expression.
  foo(auto(a));
  // - If the initializer is a braced-init-list, it shall consist of a single brace-enclosed assignment-expression and E is the assignment-expression.
  foo(auto{a});
  foo(auto({a})); // expected-error {{cannot deduce actual type for 'auto' from parenthesized initializer list}}
  foo(auto{{a}}); // expected-error {{cannot deduce actual type for 'auto' from nested initializer list}}

  foo(auto(&A::g)); // expected-error {{functional-style cast to 'auto' has incompatible initializer of type '<overloaded function type>'}}

  foo(auto(a, 3.14));     // expected-error {{initializer for functional-style cast to 'auto' contains multiple expressions}}
  foo(auto{a, 3.14});     // expected-error {{initializer for functional-style cast to 'auto' contains multiple expressions}}
  foo(auto({a, 3.14}));   // expected-error {{cannot deduce actual type for 'auto' from parenthesized initializer list}}
  foo(auto{{a, 3.14}});   // expected-error {{cannot deduce actual type for 'auto' from nested initializer list}}
  foo(auto({a}, {3.14})); // expected-error {{initializer for functional-style cast to 'auto' contains multiple expressions}}
  foo(auto{{a}, {3.14}}); // expected-error {{initializer for functional-style cast to 'auto' contains multiple expressions}}

  foo(auto{1, 2});   // expected-error {{initializer for functional-style cast to 'auto' contains multiple expressions}}
  foo(auto({1, 2})); // expected-error {{cannot deduce actual type for 'auto' from parenthesized initializer list}}
  foo(auto{{1, 2}}); // expected-error {{cannot deduce actual type for 'auto' from nested initializer list}}
}

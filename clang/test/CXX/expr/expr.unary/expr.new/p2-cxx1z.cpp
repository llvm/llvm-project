// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++14
// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++17 -pedantic

// [expr.new]p2 ... the invented declaration: T x init ;
// C++2b [dcl.type.auto.deduct]p2.2
// For a variable declared with a type that contains a placeholder type, T is the declared type of the variable.
void f() {
  // - If the initializer is a parenthesized expression-list, the expression-list shall be a single assignmentexpression and E is the assignment-expression.
  new auto('a');
  new decltype(auto)('a');
  // - If the initializer is a braced-init-list, it shall consist of a single brace-enclosed assignment-expression and E is the assignment-expression.
  new auto{2};
  new decltype(auto){2};

  new auto{};   // expected-error{{new expression for type 'auto' requires a constructor argument}}
  new auto({}); // expected-error{{cannot deduce actual type for 'auto' from parenthesized initializer list}}
  new auto{{}}; // expected-error{{cannot deduce actual type for 'auto' from nested initializer list}}

  new auto({2});  // expected-error{{cannot deduce actual type for 'auto' from parenthesized initializer list}}
  new auto{{2}};  // expected-error{{cannot deduce actual type for 'auto' from nested initializer list}}
  new auto{1, 2}; // expected-error{{new expression for type 'auto' contains multiple constructor arguments}}

  new decltype(auto){};   // expected-error{{new expression for type 'decltype(auto)' requires a constructor argument}}
  new decltype(auto)({}); // expected-error{{cannot deduce actual type for 'decltype(auto)' from parenthesized initializer list}}
  new decltype(auto){{}}; // expected-error{{cannot deduce actual type for 'decltype(auto)' from nested initializer list}}

  new decltype(auto)({1});    // expected-error{{cannot deduce actual type for 'decltype(auto)' from parenthesized initializer list}}
  new decltype(auto){1, 2};   // expected-error{{new expression for type 'decltype(auto)' contains multiple constructor arguments}}
  new decltype(auto)({1, 2}); // expected-error{{cannot deduce actual type for 'decltype(auto)' from parenthesized initializer list}}
}

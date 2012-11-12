// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

struct X { };

template<typename T> T& lvalue();
template<typename T> T&& xvalue();
template<typename T> T prvalue();

// In a .* expression whose object expression is an rvalue, the
// program is ill-formed if the second operand is a pointer to member
// function with ref-qualifier &. In a ->* expression or in a .*
// expression whose object expression is an lvalue, the program is
// ill-formed if the second operand is a pointer to member function
// with ref-qualifier &&.
void test(X *xp, int (X::*pmf)(int), int (X::*l_pmf)(int) &, 
          int (X::*r_pmf)(int) &&) {
  // No ref-qualifier.
  (lvalue<X>().*pmf)(17);
  (xvalue<X>().*pmf)(17);
  (prvalue<X>().*pmf)(17);
  (xp->*pmf)(17);

  // Lvalue ref-qualifier.
  (lvalue<X>().*l_pmf)(17);
  (xvalue<X>().*l_pmf)(17); // expected-error{{pointer-to-member function type 'int (X::*)(int) &' can only be called on an lvalue}}
  (prvalue<X>().*l_pmf)(17); // expected-error{{pointer-to-member function type 'int (X::*)(int) &' can only be called on an lvalue}}
  (xp->*l_pmf)(17);

  // Rvalue ref-qualifier.
  (lvalue<X>().*r_pmf)(17); // expected-error{{pointer-to-member function type 'int (X::*)(int) &&' can only be called on an rvalue}}
  (xvalue<X>().*r_pmf)(17);
  (prvalue<X>().*r_pmf)(17);
  (xp->*r_pmf)(17);  // expected-error{{pointer-to-member function type 'int (X::*)(int) &&' can only be called on an rvalue}}
}

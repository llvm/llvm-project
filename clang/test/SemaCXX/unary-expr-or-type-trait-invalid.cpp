// RUN: %clang_cc1 -fsyntax-only -verify %s

a() {struct b c (sizeof(b * [({ {tree->d* next)} 0

// expected-error@3 {{a type specifier is required for all declarations}}
// expected-error@3 {{use of undeclared identifier 'tree'; did you mean 'true'?}}
// expected-error@3 {{member reference type 'bool' is not a pointer}}
// expected-error@3 {{expected ';' after expression}}
// expected-error@3 {{use of undeclared identifier 'next'; did you mean 'new'?}}
// expected-error@3 {{expected expression}}
// expected-error@3 {{expected ';' after expression}}
// expected-error@26 {{expected '}'}}
// expected-note@3 {{to match this '{'}}
// expected-error@26 {{expected ')'}}
// expected-note@3 {{to match this '('}}
// expected-error@26 {{expected ']'}}
// expected-note@3 {{to match this '['}}
// expected-error@26 {{expected ')'}}
// expected-note@3 {{to match this '('}}
// expected-error@3 {{using declaration 'exp' instantiates to an empty pack}}
// expected-error@3 {{variable has incomplete type 'struct b'}}
// expected-note@3 {{forward declaration of 'b'}}
// expected-error@3 {{expected ';' at end of declaration}}
// expected-error@26 {{expected '}'}}
// expected-note@3 {{to match this '{'}}
// expected-warning@3 {{expression result unused}}

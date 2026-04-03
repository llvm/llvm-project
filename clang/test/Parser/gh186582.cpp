// RUN: %clang_cc1 -fsyntax-only -std=c++23 -verify %s

a(   ::template operator; 
// expected-error@3 {{expected a type}}
// expected-error@3 {{a type specifier is required for all declarations}}
// expected-error@3 {{expected unqualified-id}}
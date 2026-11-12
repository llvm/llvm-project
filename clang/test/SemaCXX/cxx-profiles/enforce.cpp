// RUN: %clang_cc1 -std=c++20 -fexperimental-cxx-profiles -fsyntax-only -verify %s

[[profiles::enforce(std::casting)]]; // #enforce
[[profiles::enforce(std::casting)]]; // Repeating a designator is a no-op.
[[profiles::enforce(std::casting(strict))]];
// expected-error@-1 {{profile 'std::casting' is already enforced with designator 'std::casting'}}
// expected-note@#enforce {{profile 'std::casting' was enforced here}}
[[profiles::enforce(acme::thing)]];
// expected-warning@-1 {{unknown profile 'acme::thing'; it will not be enforced}}
[[profiles::enforce]];
// expected-error@-1 {{'profiles::enforce' attribute requires at least one profile designator}}

namespace N { // #first-decl
[[profiles::enforce(std::casting)]];
// expected-error@-1 {{'profiles::enforce' attribute must appear at translation unit scope}}
}

[[profiles::enforce(std::casting)]];
// expected-error@-1 {{'profiles::enforce' attribute must precede every declaration in the translation unit}}
// expected-note@#first-decl {{first declaration is here}}

[[profiles::enforce(std::casting)]] int x;
// expected-error@-1 {{'profiles::enforce' attribute only applies to empty declarations}}

[[profiles::require(std::casting)]] int y;
// expected-error@-1 {{'profiles::require' attribute may only appear on a module import declaration}}

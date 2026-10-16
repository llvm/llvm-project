// RUN: %clang_cc1 -fsyntax-only -verify %s

// issue144264
constexpr void test() 
{ 
    using TT = struct T[; 
    // expected-error@-1 {{expected expression}}
}

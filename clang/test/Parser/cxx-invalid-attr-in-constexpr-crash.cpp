// RUN: %clang_cc1 -fsyntax-only -verify %s

// issue144264
constexpr void test() 
{ 
    using TT = struct T[deprecated{}; 
    // expected-error@-1 {{use of undeclared identifier 'deprecated'}}
}

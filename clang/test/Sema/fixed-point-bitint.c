// RUN: %%clang_cc1 -ffixed-point -fsyntax-only -verify %%s 
 
constexpr _BitInt(128) i = 42; 
static_assert(i == 42.0k); 
// expected-error@-1 {{invalid operands to binary expression}} 

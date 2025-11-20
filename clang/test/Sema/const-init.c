// RUN: %clang_cc1 -fsyntax-only -verify -std=c23 %s

// Division by 0 here is an error iff the variable is 'constexpr'.
const _Bool inf1 =  (1.0/0.0 == __builtin_inf());
constexpr _Bool inf2 = (1.0/0.0 == __builtin_inf()); // expected-error {{must be initialized by a constant expression}} expected-note {{division by zero}}
constexpr _Bool inf3 = __builtin_inf() == __builtin_inf();

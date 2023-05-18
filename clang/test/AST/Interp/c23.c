// RUN: %clang_cc1 -std=c23 -fexperimental-new-constant-interpreter -verify=expected,both %s
// RUN: %clang_cc1 -std=c23 -verify=ref,both %s



const _Bool inf1 =  (1.0/0.0 == __builtin_inf());
constexpr _Bool inf2 = (1.0/0.0 == __builtin_inf()); // both-error {{must be initialized by a constant expression}} \
                                                     // both-note {{division by zero}}
constexpr _Bool inf3 = __builtin_inf() == __builtin_inf();



// RUN: %clang_cc1 -std=c23 -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c17 -fsyntax-only -pedantic -verify=pedantic %s
// RUN: %clang_cc1 -std=c17 -fsyntax-only -Wpre-c2x-compat -verify=pre-c2x-compat %s

typedef int (*T1)[2];
restrict T1 t1;

typedef int *T2[2];
restrict T2 t2;         // pedantic-warning {{'restrict' qualifier on an array of pointers is a C23 extension}} \
                        // pre-c2x-compat-warning {{'restrict' qualifier on an array of pointers is incompatible with C standards before C23}}

typedef int *T3[2][2];
restrict T3 t3;         // pedantic-warning {{'restrict' qualifier on an array of pointers is a C23 extension}} \
                        // pre-c2x-compat-warning {{'restrict' qualifier on an array of pointers is incompatible with C standards before C23}}

typedef int (*t4)();    // pedantic-warning {{a function declaration without a prototype is deprecated in all versions of C}}
typedef t4 t5[2];
typedef t5 restrict t6; // expected-error {{pointer to function type 'int (void)' may not be 'restrict' qualified}} \
                        // pedantic-error {{pointer to function type 'int ()' may not be 'restrict' qualified}} \
                        // pre-c2x-compat-error {{pointer to function type 'int ()' may not be 'restrict' qualified}}

typedef int t7[2];
typedef t7 restrict t8; // expected-error {{restrict requires a pointer or reference ('t7' (aka 'int[2]') is invalid)}} \
                        // pedantic-error {{restrict requires a pointer or reference ('t7' (aka 'int[2]') is invalid)}} \
                        // pre-c2x-compat-error {{restrict requires a pointer or reference ('t7' (aka 'int[2]') is invalid}}

// RUN: rm -fR %t
// RUN: split-file %s %t
// RUN: %clang_cc1 -fsyntax-only -verify -Werror=implicit-function-declaration -std=c89 %t/stdargneeds0.c
// RUN: %clang_cc1 -fsyntax-only -verify=expected-modules -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -Werror=implicit-function-declaration -std=c89 %t/stdargneeds0.c
// RUN: %clang_cc1 -fsyntax-only -verify -Werror=implicit-function-declaration -std=c89 %t/stdargneeds1.c
// RUN: %clang_cc1 -fsyntax-only -verify=expected-modules -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -Werror=implicit-function-declaration -std=c89 %t/stdargneeds1.c
// RUN: %clang_cc1 -fsyntax-only -verify -Werror=implicit-function-declaration -std=c89 %t/stdargneeds2.c
// RUN: %clang_cc1 -fsyntax-only -verify=expected-modules -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -Werror=implicit-function-declaration -std=c89 %t/stdargneeds2.c
// RUN: %clang_cc1 -fsyntax-only -verify -Werror=implicit-function-declaration -std=c89 %t/stdargneeds3.c
// RUN: %clang_cc1 -fsyntax-only -verify=expected-modules -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -Werror=implicit-function-declaration -std=c89 %t/stdargneeds3.c
// RUN: %clang_cc1 -fsyntax-only -verify -Werror=implicit-function-declaration -std=c89 %t/stdargneeds4.c
// RUN: %clang_cc1 -fsyntax-only -verify=expected-modules -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -Werror=implicit-function-declaration -std=c89 %t/stdargneeds4.c
// RUN: %clang_cc1 -fsyntax-only -verify -Werror=implicit-function-declaration -std=c89 %t/stdargneeds5.c
// RUN: %clang_cc1 -fsyntax-only -verify=expected-modules -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -Werror=implicit-function-declaration -std=c89 %t/stdargneeds5.c

// Split the file so that the "implicitly declaring library function" errors get repeated.
// Use C89 to verify that __need_ can be used to get types that wouldn't normally be available.

//--- stdargneeds0.c
static void f(int p, ...) {
    __gnuc_va_list g; // expected-error{{undeclared identifier '__gnuc_va_list'}} expected-modules-error{{undeclared identifier}}
    va_list v; // expected-error{{undeclared identifier 'va_list'}} expected-modules-error{{undeclared identifier}}
    va_start(v, p); // expected-error{{implicitly declaring library function 'va_start'}} expected-note{{include the header <stdarg.h> or explicitly provide a declaration for 'va_start'}} expected-error{{undeclared identifier 'v'}} \
                       expected-modules-error{{implicitly declaring library function}} expected-modules-note{{provide a declaration}} expected-modules-error{{undeclared identifier}}
    int i = va_arg(v, int); // expected-error{{implicit declaration of function 'va_arg'}} expected-error{{expected expression}} expected-error{{use of undeclared identifier 'v'}} \
                               expected-modules-error{{implicit declaration of function}} expected-modules-error{{expected expression}} expected-modules-error{{undeclared identifier}}
    va_end(v); // expected-error{{implicitly declaring library function 'va_end'}} expected-note{{include the header <stdarg.h> or explicitly provide a declaration for 'va_end'}} expected-error{{undeclared identifier 'v'}} \
                  expected-modules-error{{implicitly declaring library function}} expected-modules-note{{provide a declaration}} expected-modules-error{{undeclared identifier}}
    __va_copy(g, v); // expected-error{{implicit declaration of function '__va_copy'}} expected-error{{use of undeclared identifier 'g'}} expected-error{{use of undeclared identifier 'v'}} \
                        expected-modules-error{{implicit declaration of function}} expected-modules-error{{undeclared identifier}} expected-modules-error{{undeclared identifier}}
    va_copy(g, v); // expected-error{{implicitly declaring library function 'va_copy'}} expected-note{{include the header <stdarg.h> or explicitly provide a declaration for 'va_copy'}} expected-error{{use of undeclared identifier 'g'}} expected-error{{use of undeclared identifier 'v'}} \
                      expected-modules-error{{implicitly declaring library function}} expected-modules-note{{provide a declaration}} expected-modules-error{{undeclared identifier}} expected-modules-error{{undeclared identifier}}
}

//--- stdargneeds1.c
#define __need___va_list
#include <stdarg.h>
static void f(int p, ...) {
    __gnuc_va_list g;
    va_list v; // expected-error{{undeclared identifier}} \
                  expected-modules-error{{'va_list' must be declared before it is used}} expected-modules-note@__stdarg_va_list.h:*{{declaration here is not visible}}
    va_start(v, p); // expected-error{{implicitly declaring library function}} expected-note{{provide a declaration}} expected-error{{undeclared identifier}} \
                       expected-modules-error{{implicitly declaring library function}} expected-modules-note{{provide a declaration}}
    int i = va_arg(v, int); // expected-error{{implicit declaration of function}} expected-error{{expected expression}} expected-error{{undeclared identifier}} \
                               expected-modules-error{{implicit declaration of function}} expected-modules-error{{expected expression}}
    va_end(v); // expected-error{{implicitly declaring library function}} expected-note{{provide a declaration}} expected-error{{undeclared identifier}} \
                  expected-modules-error{{implicitly declaring library function}} expected-modules-note{{provide a declaration}}
    __va_copy(g, v); // expected-error{{implicit declaration of function}} expected-error{{undeclared identifier}} \
                        expected-modules-error{{implicit declaration of function}}
    va_copy(g, v); // expected-error{{implicitly declaring library function}} expected-note{{provide a declaration}} expected-error{{undeclared identifier}} \
                      expected-modules-error{{implicitly declaring library function}} expected-modules-note{{provide a declaration}}
}

//--- stdargneeds2.c
#define __need_va_list
#include <stdarg.h>
static void f(int p, ...) {
    __gnuc_va_list g; // expected-error{{undeclared identifier}} \
                         expected-modules-error{{'__gnuc_va_list' must be declared before it is used}} expected-modules-note@__stdarg___gnuc_va_list.h:*{{declaration here is not visible}}
    va_list v;
    va_start(v, p); // expected-error{{implicitly declaring library function}} expected-note{{provide a declaration}} \
                       expected-modules-error{{implicitly declaring library function}} expected-modules-note{{provide a declaration}}
    int i = va_arg(v, int); // expected-error{{implicit declaration of function}} expected-error{{expected expression}} \
                               expected-modules-error{{implicit declaration of function}} expected-modules-error{{expected expression}}
    va_end(v); // expected-error{{implicitly declaring library function}} expected-note{{provide a declaration}} \
                  expected-modules-error{{implicitly declaring library function}} expected-modules-note{{provide a declaration}}
    __va_copy(g, v); // expected-error{{implicit declaration of function}} expected-error{{undeclared identifier}} \
                        expected-modules-error{{implicit declaration of function}}
    va_copy(g, v); // expected-error{{implicitly declaring library function}} expected-note{{provide a declaration}} expected-error{{undeclared identifier}} \
                      expected-modules-error{{implicitly declaring library function}} expected-modules-note{{provide a declaration}}
}

//--- stdargneeds3.c
#define __need_va_list
#define __need_va_arg
#include <stdarg.h>
static void f(int p, ...) {
    __gnuc_va_list g; // expected-error{{undeclared identifier}} \
                         expected-modules-error{{'__gnuc_va_list' must be declared before it is used}} expected-modules-note@__stdarg___gnuc_va_list.h:*{{declaration here is not visible}}
    va_list v;
    va_start(v, p);
    int i = va_arg(v, int);
    va_end(v);
    __va_copy(g, v); // expected-error{{implicit declaration of function}} expected-error{{undeclared identifier}} \
                        expected-modules-error{{implicit declaration of function}}
    va_copy(g, v); // expected-error{{implicitly declaring library function}} expected-note{{provide a declaration}} expected-error{{undeclared identifier}} \
                      expected-modules-error{{implicitly declaring library function}} expected-modules-note{{provide a declaration}}
}

//--- stdargneeds4.c
#define __need___va_list
#define __need_va_list
#define __need___va_copy
#include <stdarg.h>
static void f(int p, ...) {
    __gnuc_va_list g;
    va_list v;
    va_start(v, p); // expected-error{{implicitly declaring library function}} expected-note{{provide a declaration}} \
                       expected-modules-error{{implicitly declaring library function}} expected-modules-note{{provide a declaration}}
    int i = va_arg(v, int); // expected-error{{implicit declaration of function}} expected-error{{expected expression}} \
                               expected-modules-error{{implicit declaration of function}} expected-modules-error{{expected expression}}
    va_end(v); // expected-error{{implicitly declaring library function}} expected-note{{provide a declaration}} \
                  expected-modules-error{{implicitly declaring library function}} expected-modules-note{{provide a declaration}}
    __va_copy(g, v);
    va_copy(g, v); // expected-error{{implicitly declaring library function}} expected-note{{provide a declaration}} \
                      expected-modules-error{{implicitly declaring library function}} expected-modules-note{{provide a declaration}}
}

//--- stdargneeds5.c
#define __need___va_list
#define __need_va_list
#define __need_va_copy
#include <stdarg.h>
static void f(int p, ...) {
    __gnuc_va_list g;
    va_list v;
    va_start(v, p); // expected-error{{implicitly declaring library function}} expected-note{{provide a declaration}} \
                       expected-modules-error{{implicitly declaring library function}} expected-modules-note{{provide a declaration}}
    int i = va_arg(v, int); // expected-error{{implicit declaration of function}} expected-error{{expected expression}} \
                               expected-modules-error{{implicit declaration of function}} expected-modules-error{{expected expression}}
    va_end(v); // expected-error{{implicitly declaring library function}} expected-note{{provide a declaration}} \
                  expected-modules-error{{implicitly declaring library function}} expected-modules-note{{provide a declaration}}
    __va_copy(g, v); // expected-error{{implicit declaration of function}} expected-modules-error{{implicit declaration of function}}
    va_copy(g, v);
}

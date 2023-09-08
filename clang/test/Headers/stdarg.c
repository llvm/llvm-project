// RUN: split-file %s %t
// RUN: %clang_cc1 -fsyntax-only -verify=c89 -Werror=implicit-function-declaration -std=c89 %t/stdarg0.c
// RUN: %clang_cc1 -fsyntax-only -verify=c99 -Werror=implicit-function-declaration -std=c99 %t/stdarg0.c
// RUN: %clang_cc1 -fsyntax-only -verify=c89 -Werror=implicit-function-declaration -std=c89 %t/stdarg1.c
// RUN: %clang_cc1 -fsyntax-only -verify=c99 -Werror=implicit-function-declaration -std=c99 %t/stdarg1.c

// Split the file so that the "implicitly declaring library function" errors get repeated.

//--- stdarg0.c
static void f(int p, ...) {
    __gnuc_va_list g; // c89-error{{undeclared identifier '__gnuc_va_list'}} c99-error{{undeclared identifier}}
    va_list v; // c89-error{{undeclared identifier 'va_list'}} c99-error{{undeclared identifier}}
    va_start(v, p); // c89-error{{implicitly declaring library function 'va_start'}} c89-note{{include the header <stdarg.h> or explicitly provide a declaration for 'va_start'}} c89-error{{undeclared identifier 'v'}} \
                       c99-error{{call to undeclared library function 'va_start'}} c99-note{{provide a declaration}} c99-error{{undeclared identifier}}
    int i = va_arg(v, int); // c89-error{{implicit declaration of function 'va_arg'}} c89-error{{expected expression}} c89-error{{use of undeclared identifier 'v'}} \
                               c99-error{{call to undeclared function 'va_arg'}} c99-error{{expected expression}} c99-error{{undeclared identifier}}
    va_end(v); // c89-error{{implicitly declaring library function 'va_end'}} c89-note{{include the header <stdarg.h> or explicitly provide a declaration for 'va_end'}} c89-error{{undeclared identifier 'v'}} \
                  c99-error{{call to undeclared library function 'va_end'}} c99-note{{provide a declaration}} c99-error{{undeclared identifier}}
    __va_copy(g, v); // c89-error{{implicit declaration of function '__va_copy'}} c89-error{{use of undeclared identifier 'g'}} c89-error{{use of undeclared identifier 'v'}} \
                        c99-error{{call to undeclared function '__va_copy'}} c99-error{{undeclared identifier}} c99-error{{undeclared identifier}}
    va_copy(g, v); // c89-error{{implicitly declaring library function 'va_copy'}} c89-note{{include the header <stdarg.h> or explicitly provide a declaration for 'va_copy'}} c89-error{{use of undeclared identifier 'g'}} c89-error{{use of undeclared identifier 'v'}} \
                      c99-error{{call to undeclared library function 'va_copy'}} c99-note{{provide a declaration}} c99-error{{undeclared identifier}} c99-error{{undeclared identifier}}
}

//--- stdarg1.c
// c99-no-diagnostics

#include <stdarg.h>
static void f(int p, ...) {
    __gnuc_va_list g;
    va_list v;
    va_start(v, p);
    int i = va_arg(v, int);
    va_end(v);
    __va_copy(g, v);
    va_copy(g, v); // c89-error{{implicitly declaring library function}} c89-note{{provide a declaration}}
}

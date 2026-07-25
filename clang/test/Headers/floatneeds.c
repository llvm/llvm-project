// RUN: rm -fR %t
// RUN: split-file %s %t
// RUN: %clang_cc1 -fsyntax-only -verify=c99 -std=c99 %t/floatneeds0.c
// RUN: %clang_cc1 -fsyntax-only -verify=c99 -std=c99 %t/floatneeds1.c
// RUN: %clang_cc1 -fsyntax-only -verify=c23 -std=c23 %t/floatneeds0.c
// RUN: %clang_cc1 -fsyntax-only -verify=c23 -std=c23 %t/floatneeds1.c
// RUN: %clang_cc1 -fsyntax-only -verify=c99-modules -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -std=c99 %t/floatneeds0.c
// RUN: %clang_cc1 -fsyntax-only -verify=c99-modules -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -std=c99 %t/floatneeds1.c
// RUN: %clang_cc1 -fsyntax-only -verify=c23-modules -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -std=c23 %t/floatneeds0.c
// RUN: %clang_cc1 -fsyntax-only -verify=c23-modules -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -std=c23 %t/floatneeds1.c

// Use C99 to verify that __need_ can be used to get types that wouldn't normally be available.

//--- floatneeds0.c
float infinity0 = INFINITY; // c99-error{{undeclared identifier 'INFINITY'}} c23-error{{undeclared identifier 'INFINITY'}} \
                               c99-modules-error{{undeclared identifier 'INFINITY'}} c23-modules-error{{undeclared identifier 'INFINITY'}}
float nan0 = NAN; // c99-error{{undeclared identifier 'NAN'}} c23-error{{undeclared identifier 'NAN'}} \
                     c99-modules-error{{undeclared identifier 'NAN'}} c23-modules-error{{undeclared identifier 'NAN'}}
float max0 = FLT_MAX; // c99-error{{undeclared identifier 'FLT_MAX'}} c23-error{{undeclared identifier 'FLT_MAX'}} \
                         c99-modules-error{{undeclared identifier 'FLT_MAX'}} c23-modules-error{{undeclared identifier 'FLT_MAX'}}

#define __need_infinity_nan
#include <float.h>
float infinity1 = INFINITY;
float nan1 = NAN;
float max1 = FLT_MAX; // c99-error{{undeclared identifier}} c23-error{{undeclared identifier}} \
                         c99-modules-error{{undeclared identifier}} c23-modules-error{{undeclared identifier}}

#include <float.h>
float infinity2 = INFINITY;
float nan2 = NAN;
float max2 = FLT_MAX;

//--- floatneeds1.c
// c23-no-diagnostics
// c23-modules-no-diagnostics

#include <float.h>
float infinity0 = INFINITY; // c99-error{{undeclared identifier}} c99-modules-error{{undeclared identifier}}
float nan0 = NAN; // c99-error{{undeclared identifier}} c99-modules-error{{undeclared identifier}}
float max0 = FLT_MAX;

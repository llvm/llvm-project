// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: echo 'struct X {}; struct Y : X { friend int f(Y); };' > %t/a.h
// RUN: echo 'module a { header "a.h" }' > %t/map
// RUN: %clang_cc1 -fmodules -x c++ -emit-module -fmodule-name=a %t/map -o %t/a.pcm
// RUN: %clang_cc1 -fmodules -x c++ -verify -fmodule-file=%t/a.pcm %s -fno-modules-error-recovery

struct X;
struct Y;

// Ensure that we can't use the definitions of X and Y, since we've not imported module a.
Y *yp;
X *xp = yp; // expected-error {{cannot initialize}}
_Static_assert(!__is_convertible(Y*, X*), "");
X &xr = *yp; // expected-error {{unrelated type}}
int g(Y &y) { f(y); } // expected-error {{undeclared identifier 'f'}}

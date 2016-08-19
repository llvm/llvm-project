// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: echo 'int a, b;' > %t/x.h
// RUN: echo 'module x { header "x.h" module y {} } module z {}' > %t/map
// RUN: %clang_cc1 -std=c++1z -fmodules-ts -I%S/Inputs -fmodules-cache-path=%t -fmodule-map-file=%t/map -verify %s \
// RUN:            -DTEST=1 -DMODULE_KIND=implementation -DMODULE_NAME=z
// RUN: %clang_cc1 -std=c++1z -fmodules-ts -I%S/Inputs -fmodules-cache-path=%t -fmodule-map-file=%t/map -verify %s \
// RUN:            -DTEST=2 -DMODULE_KIND=implementation -DMODULE_NAME=x
// RUN: %clang_cc1 -std=c++1z -fmodules-ts -I%S/Inputs -fmodules-cache-path=%t -fmodule-map-file=%t/map -verify %s \
// RUN:            -DTEST=3 -DMODULE_KIND= -DMODULE_NAME=z
// RUN: %clang_cc1 -std=c++1z -fmodules-ts -I%S/Inputs -fmodules-cache-path=%t -fmodule-map-file=%t/map -verify %s \
// RUN:            -DTEST=4 -DMODULE_KIND=partition -DMODULE_NAME=z
// RUN: %clang_cc1 -std=c++1z -fmodules-ts -I%S/Inputs -fmodules-cache-path=%t -fmodule-map-file=%t/map -verify %s \
// RUN:            -DTEST=5 -DMODULE_KIND=elderberry -DMODULE_NAME=z
// RUN: %clang_cc1 -std=c++1z -fmodules-ts -I%S/Inputs -fmodules-cache-path=%t -fmodule-map-file=%t/map -verify %s \
// RUN:            -DTEST=1 -DMODULE_KIND=implementation -DMODULE_NAME='z [[]]'
// RUN: %clang_cc1 -std=c++1z -fmodules-ts -I%S/Inputs -fmodules-cache-path=%t -fmodule-map-file=%t/map -verify %s \
// RUN:            -DTEST=6 -DMODULE_KIND=implementation -DMODULE_NAME='z [[fancy]]'
// RUN: %clang_cc1 -std=c++1z -fmodules-ts -I%S/Inputs -fmodules-cache-path=%t -fmodule-map-file=%t/map -verify %s \
// RUN:            -DTEST=7 -DMODULE_KIND=implementation -DMODULE_NAME='z [[maybe_unused]]'

module MODULE_KIND MODULE_NAME;
#if TEST == 3
// expected-error@-2 {{'module' declaration found while not building module interface}}
#elif TEST == 4
// expected-error@-4 {{'module partition' declaration found while not building module interface}}
#elif TEST == 5
// expected-error@-6 {{unexpected module kind 'elderberry'}}
#elif TEST == 6
// expected-warning@-8 {{unknown attribute 'fancy' ignored}}
#elif TEST == 7
// expected-error-re@-10 {{'maybe_unused' attribute cannot be applied to a module{{$}}}}
#endif

int use_1 = a;
#if TEST != 2
// expected-error@-2 {{undeclared}}
#endif

import x;

int use_2 = b; // ok

import x [[]];
import x [[foo]]; // expected-warning {{unknown attribute 'foo' ignored}}
import x [[noreturn]]; // expected-error {{'noreturn' attribute cannot be applied to a module import}}
import x [[blarg::noreturn]]; // expected-warning {{unknown attribute 'noreturn' ignored}}

import x.y;
import x.; // expected-error {{expected a module name after module import}}
import .x; // expected-error {{expected a module name after module import}}

import blarg; // expected-error {{module 'blarg' not found}}

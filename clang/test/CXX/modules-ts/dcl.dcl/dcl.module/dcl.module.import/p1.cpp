// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: echo 'export module x; export int a, b;' > %t/x.cppm
// RUN: echo 'export module x.y; export int c;' > %t/x.y.cppm
// RUN: echo 'export module a.b; export int d;' > %t/a.b.cppm
//
// RUN: %clang_cc1 -std=c++1z -fmodules-ts -emit-module-interface %t/x.cppm -o %t/x.pcm
// RUN: %clang_cc1 -std=c++1z -fmodules-ts -emit-module-interface -fmodule-file=%t/x.pcm %t/x.y.cppm -o %t/x.y.pcm
// RUN: %clang_cc1 -std=c++1z -fmodules-ts -emit-module-interface %t/a.b.cppm -o %t/a.b.pcm
//
// RUN: %clang_cc1 -std=c++1z -fmodules-ts -I%t -fmodule-file=%t/x.y.pcm -fmodule-file=%t/a.b.pcm -verify %s \
// RUN:            -DMODULE_NAME=z -DINTERFACE
// RUN: %clang_cc1 -std=c++1z -fmodules-ts -I%t -fmodule-file=%t/x.y.pcm -fmodule-file=%t/a.b.pcm -verify %s \
// RUN:            -DMODULE_NAME=a.b
// RUN: %clang_cc1 -std=c++1z -fmodules-ts -I%t -fmodule-file=%t/x.y.pcm -fmodule-file=%t/a.b.pcm -verify %s \
// RUN:            -DMODULE_X -DMODULE_NAME=x

#ifdef INTERFACE
export
#endif
module MODULE_NAME;

int use_1 = a;
#if !MODULE_X
// expected-error@-2 {{declaration of 'a' must be imported from module 'x' before it is required}}
// expected-note@x.cppm:1 {{here}}
#endif

import x;

int use_2 = b; // ok

// There is no relation between module x and module x.y.
int use_3 = c; // expected-error {{declaration of 'c' must be imported from module 'x.y'}}
               // expected-note@x.y.cppm:1 {{here}}

import x [[]];
import x [[foo]]; // expected-warning {{unknown attribute 'foo' ignored}}
import x [[noreturn]]; // expected-error {{'noreturn' attribute cannot be applied to a module import}}
import x [[blarg::noreturn]]; // expected-warning {{unknown attribute 'noreturn' ignored}}

import x.y;
import a.b; // Does not imply existence of module a.
import x.; // expected-error {{expected a module name after 'import'}}
import .x; // expected-error {{expected a module name after 'import'}}

int use_4 = c; // ok

import blarg; // expected-error {{module 'blarg' not found}}

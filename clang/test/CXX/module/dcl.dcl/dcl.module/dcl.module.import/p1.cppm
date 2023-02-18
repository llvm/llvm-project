// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/x.cppm -o %t/x.pcm
// RUN: %clang_cc1 -std=c++20 -emit-module-interface -fmodule-file=%t/x.pcm %t/x.y.cppm -o %t/x.y.pcm
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/a.b.cppm -o %t/a.b.pcm
//
// RUN: %clang_cc1 -std=c++20 -I%t -fmodule-file=%t/x.y.pcm -fmodule-file=%t/a.b.pcm -verify %t/test.cpp \
// RUN:            -DMODULE_NAME=z -DINTERFACE
// RUN: %clang_cc1 -std=c++20 -I%t -fmodule-file=%t/x.y.pcm -fmodule-file=%t/a.b.pcm -verify %t/test.cpp \
// RUN:            -DMODULE_NAME=a.b
// RUN: %clang_cc1 -std=c++20 -I%t -fmodule-file=%t/x.y.pcm -fmodule-file=%t/a.b.pcm -verify %t/test.x.cpp

//--- x.cppm
export module x;
export int a, b;

//--- x.y.cppm
export module x.y;
export int c;

//--- a.b.cppm
export module a.b;
export int d;

//--- test.x.cpp
module x;
int use_1 = a; // ok

int use_2 = b; // ok

// There is no relation between module x and module x.y.
int use_3 = c; // expected-error {{declaration of 'c' must be imported from module 'x.y'}}
               // expected-note@x.y.cppm:2 {{not visible}}

//--- test.cpp
#ifdef INTERFACE
export module MODULE_NAME;
#else
module MODULE_NAME;
#endif

import x;

import x [[]];
import x [[foo]]; // expected-warning {{unknown attribute 'foo' ignored}}
import x [[noreturn]]; // expected-error {{'noreturn' attribute cannot be applied to a module import}}
import x [[blarg::noreturn]]; // expected-warning {{unknown attribute 'noreturn' ignored}}

import x.y;
import x.; // expected-error {{expected a module name after 'import'}}
import .x; // expected-error {{expected a module name after 'import'}}

import blarg; // expected-error {{module 'blarg' not found}}

int use_4 = c; // ok

// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -std=c++20 %t/A.cppm -triple x86_64-linux-gnu -verify
// RUN: %clang_cc1 -std=c++20 %t/B.cppm -triple x86_64-linux-gnu -verify
// RUN: %clang_cc1 -std=c++20 %t/C.cppm -triple x86_64-linux-gnu -verify
// RUN: %clang_cc1 -std=c++20 %t/D.cppm -triple x86_64-linux-gnu -verify
// RUN: %clang_cc1 -std=c++20 %t/E.cppm -triple x86_64-linux-gnu -verify
// RUN: %clang_cc1 -std=c++20 %t/F.cppm -triple x86_64-linux-gnu -verify
// RUN: %clang_cc1 -std=c++20 %t/G.cppm -triple x86_64-linux-gnu -verify
// RUN: %clang_cc1 -std=c++20 %t/H.cppm -triple x86_64-linux-gnu -verify
// RUN: %clang_cc1 -std=c++20 %t/I.cppm -triple x86_64-linux-gnu -verify
// RUN: %clang_cc1 -std=c++20 %t/J.cppm -triple x86_64-linux-gnu -verify

//--- version.h
#ifndef VERSION_H
#define VERSION_H

#define VERSION libv5
#define A a
#define B b
#define C c
#define FUNC_LIKE(X) function_like_##X
#define ATTRS [[]]
#define SEMICOLON ;

#endif // VERSION_H

//--- A.cppm
module;
#include "version.h"
export module VERSION;  // expected-error {{the module name in a module declaration cannot contain an object-like macro 'VERSION'}}

//--- B.cppm
module;
#include "version.h"
export module A.B;      // expected-error {{the module name in a module declaration cannot contain an object-like macro 'A'}} \
                        // expected-error {{the module name in a module declaration cannot contain an object-like macro 'B'}}

//--- C.cppm
module;                             // expected-error {{missing 'module' declaration at end of global module fragment introduced here}}
#include "version.h"
export module A.FUNC_LIKE(foo):C;   // expected-error {{the module name in a module declaration cannot contain an object-like macro 'A'}} \
                                    // expected-error {{unexpected '(' after the module name in a module declaration}}

//--- D.cppm
module;                               // expected-error {{missing 'module' declaration at end of global module fragment introduced here}}
#include "version.h"
export module B.A.FUNC_LIKE(bar):C;   // expected-error {{the module name in a module declaration cannot contain an object-like macro 'B'}} \
                                      // expected-error {{the module name in a module declaration cannot contain an object-like macro 'A'}} \
                                      // expected-error {{unexpected '(' after the module name in a module declaration}}

//--- E.cppm
module;
#include "version.h"
export module a.FUNC_LIKE:c; // OK, FUNC_LIKE would not be treated as a macro name.
// expected-no-diagnostics

//--- F.cppm
module;
#include "version.h"
export module a.FUNC_LIKE:c ATTRS; // OK, FUNC_LIKE would not be treated as a macro name.
// expected-no-diagnostics

//--- G.cppm
module;                               // expected-error {{missing 'module' declaration at end of global module fragment introduced here}}
#include "version.h"
export module A.FUNC_LIKE(B c:C ATTRS // expected-error {{the module name in a module declaration cannot contain an object-like macro 'A'}} \
                                      // expected-error {{unexpected '(' after the module name in a module declaration}}

//--- H.cppm
module;                                   // expected-error {{missing 'module' declaration at end of global module fragment introduced here}}
#include "version.h"
export module A.FUNC_LIKE(B,). c:C ATTRS  // expected-error {{the module name in a module declaration cannot contain an object-like macro 'A'}} \
                                          // expected-error {{unexpected '(' after the module name in a module declaration}}

//--- I.cppm
module;                                   // expected-error {{missing 'module' declaration at end of global module fragment introduced here}}
#include "version.h"
export module A.FUNC_LIKE(B,) c:C ATTRS   // expected-error {{the module name in a module declaration cannot contain an object-like macro 'A'}} \
                                          // expected-error {{unexpected '(' after the module name in a module declaration}}

//--- J.cppm
module;
#include "version.h"
export module unexpanded : unexpanded ATTRS SEMICOLON // OK, ATTRS and SEMICOLON can be expanded.
// expected-no-diagnostics

// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/A.cppm -triple x86_64-linux-gnu -verify
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/B.cppm -triple x86_64-linux-gnu -verify
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/C.cppm -triple x86_64-linux-gnu -verify
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/D.cppm -triple x86_64-linux-gnu -verify

//--- version.h
#ifndef VERSION_H
#define VERSION_H

#define VERSION libv5
#define A a
#define B b
#define C c
#define FUNC_LIKE(X) function_like_##X

#endif

//--- A.cppm
export module x;
#include "version.h"
export module VERSION;  // expected-error {{the name of a module declaration cannot contains an object-like macro 'VERSION', and the macro will not expand}}

//--- B.cppm
export module x;
#include "version.h"
export module A.B;      // expected-error {{the name of a module declaration cannot contains an object-like macro 'A', and the macro will not expand}} \
                        // expected-error {{the name of a module declaration cannot contains an object-like macro 'B', and the macro will not expand}}

//--- C.cppm
export module x;
#include "version.h"
export module A.FUNC_LIKE(foo):C;   // expected-error {{the name of a module declaration cannot contains an object-like macro 'A', and the macro will not expand}} \
                                    // expected-error {{the name of a module declaration cannot contains a function-like macro 'FUNC_LIKE', and the macro will not expand}} \
                                    // expected-error {{the name of a module partition declaration cannot contains an object-like macro 'C', and the macro will not expand}}

//--- D.cppm
export module x;
#include "version.h"
export module B.A.FUNC_LIKE(bar):C;   // expected-error {{the name of a module declaration cannot contains an object-like macro 'B', and the macro will not expand}} \
                                      // expected-error {{the name of a module declaration cannot contains an object-like macro 'A', and the macro will not expand}} \
                                      // expected-error {{the name of a module declaration cannot contains a function-like macro 'FUNC_LIKE', and the macro will not expand}} \
                                      // expected-error {{the name of a module partition declaration cannot contains an object-like macro 'C', and the macro will not expand}}

//--- E.cppm
export module x;
#include "version.h"
export module a.FUNC_LIKE:c // OK, FUNC_LIKE would not be treated as a macro name.
// expected-no-diagnostics
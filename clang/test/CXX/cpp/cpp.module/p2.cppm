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

//--- version.h
#ifndef VERSION_H
#define VERSION_H

#define VERSION libv5
#define A a
#define B b
#define C c
#define FUNC_LIKE(X) function_like_##X
#define ATTR [[]]

#endif

//--- A.cppm
#include "version.h"
export module VERSION;  // expected-error {{the name of a module declaration cannot contains an object-like macro 'VERSION', and the macro will not expand}}

//--- B.cppm
#include "version.h"
export module A.B;      // expected-error {{the name of a module declaration cannot contains an object-like macro 'A', and the macro will not expand}} \
                        // expected-error {{the name of a module declaration cannot contains an object-like macro 'B', and the macro will not expand}}

//--- C.cppm
#include "version.h"
export module A.FUNC_LIKE(foo):C;   // expected-error {{the name of a module declaration cannot contains an object-like macro 'A', and the macro will not expand; did you mean 'a'?}} \
                                    // expected-error {{unexpected '(' after the name of a module declaration}} \
                                    // expected-error {{the name of a module partition declaration cannot contains an object-like macro 'C', and the macro will not expand}}

//--- D.cppm
#include "version.h"
export module B.A.FUNC_LIKE(bar):C;   // expected-error {{the name of a module declaration cannot contains an object-like macro 'B', and the macro will not expand; did you mean 'b'?}} \
                                      // expected-error {{the name of a module declaration cannot contains an object-like macro 'A', and the macro will not expand; did you mean 'a'?}} \
                                      // expected-error {{unexpected '(' after the name of a module declaration}} \
                                      // expected-error {{the name of a module partition declaration cannot contains an object-like macro 'C', and the macro will not expand; did you mean 'c'?}}

//--- E.cppm
#include "version.h"
export module a.FUNC_LIKE:c; // OK, FUNC_LIKE would not be treated as a macro name.
// expected-no-diagnostics

//--- F.cppm
#include "version.h"
export module a.FUNC_LIKE:c ATTR; // OK, FUNC_LIKE would not be treated as a macro name.
// expected-no-diagnostics

//--- G.cppm
#include "version.h"
export module A.FUNC_LIKE(B c:C ATTR  // expected-error {{the name of a module declaration cannot contains an object-like macro 'A', and the macro will not expand; did you mean 'a'?}} \
                                      // expected-error {{unexpected '(' after the name of a module declaration}} \
                                      // expected-error {{the name of a module partition declaration cannot contains an object-like macro 'C', and the macro will not expand; did you mean 'c'?}} \
                                      // expected-error {{expected ';' after module name}}

//--- H.cppm
#include "version.h"
export module A.FUNC_LIKE(B,). c:C ATTR   // expected-error {{the name of a module declaration cannot contains an object-like macro 'A', and the macro will not expand; did you mean 'a'?}} \
                                          // expected-error {{unexpected '(' after the name of a module declaration}} \
                                          // expected-error {{the name of a module partition declaration cannot contains an object-like macro 'C', and the macro will not expand; did you mean 'c'?}} \
                                          // expected-error {{expected ';' after module name}}

//--- I.cppm
#include "version.h"
export module A.FUNC_LIKE(B,) c:C ATTR    // expected-error {{the name of a module declaration cannot contains an object-like macro 'A', and the macro will not expand; did you mean 'a'?}} \
                                          // expected-error {{unexpected '(' after the name of a module declaration}} \
                                          // expected-error {{expected ';' after module name}} \
                                          // expected-error {{unknown type name 'c'}} \
                                          // expected-error {{expected unqualified-id}}

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
export module VERSION;  // expected-error {{the name of a module declaration cannot contains an object-like macro 'VERSION'}}

//--- B.cppm
export module x;
#include "version.h"
export module A.B;      // expected-error {{the name of a module declaration cannot contains an object-like macro 'A'}} \
                        // expected-error {{the name of a module declaration cannot contains an object-like macro 'B'}}

//--- C.cppm
export module x;
#include "version.h"
export module A.FUNC_LIKE(foo):C;   // expected-error {{the name of a module declaration cannot contains an object-like macro 'A'}} \
                                    // expected-error {{the name of a module declaration cannot contains a function-like macro 'FUNC_LIKE'}} \
                                    // expected-error {{the name of a module partition declaration cannot contains an object-like macro 'C'}}

//--- D.cppm
export module x;
#include "version.h"
export module B.A.FUNC_LIKE(bar):C;   // expected-error {{the name of a module declaration cannot contains an object-like macro 'B'}} \
                                      // expected-error {{the name of a module declaration cannot contains an object-like macro 'A'}} \
                                      // expected-error {{the name of a module declaration cannot contains a function-like macro 'FUNC_LIKE'}} \
                                      // expected-error {{the name of a module partition declaration cannot contains an object-like macro 'C'}}

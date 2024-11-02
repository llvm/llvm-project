// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
// RUN: cd %t
//
// RUN: %clang_cc1 -std=c++20 M.cpp -fsyntax-only -DTEST_INTERFACE -verify
// RUN: %clang_cc1 -std=c++20 M.cpp -emit-module-interface -o M.pcm
// RUN: %clang_cc1 -std=c++20 useM.cpp -fsyntax-only -fmodule-file=M.pcm -verify

//--- decls.h
int f(); // #1, attached to the global module
int g(); // #2, attached to the global module

//--- M.cpp
module;
#include "decls.h"
export module M;
export using ::f; // OK, does not declare an entity, exports #1
#if TEST_INTERFACE
// error: matches #2, but attached to M
int g(); // expected-error {{declaration of 'g' in module M follows declaration in the global module}}
// expected-note@decls.h:2 {{previous declaration is here}}
#endif
export int h(); // #3
export int k(); // #4

//--- useM.cpp
import M;
// error: matches #3
static int h(); // expected-error {{static declaration of 'h' follows non-static declaration}}
// expected-note@M.cpp:10 {{previous declaration is here}}

// error: matches #4
int k(); // expected-error {{declaration of 'k' in the global module follows declaration in module M}}
// expected-note@M.cpp:11 {{previous declaration is here}}

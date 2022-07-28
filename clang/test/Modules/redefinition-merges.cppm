// Tests that redefinitions in different TUs could be merged correctly and the
// redefinitions in the same TUs could be merged diagnosticed correctly.
//
// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 -I%t %t/normal.cpp -verify -fsyntax-only
// RUN: %clang_cc1 -std=c++20 -I%t %t/M1.cppm -verify -fsyntax-only
// RUN: %clang_cc1 -std=c++20 -I%t %t/M2.cppm -verify -fsyntax-only
// RUN: %clang_cc1 -std=c++20 -I%t %t/M3.cppm -verify -fsyntax-only
// RUN: %clang_cc1 -std=c++20 -I%t %t/M.cppm -emit-module-interface -o %t/M.pcm
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t %t/Use1.cpp -verify -fsyntax-only
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t %t/Use2.cpp -verify -fsyntax-only
//
//--- foo.h
#ifndef FOO
#define FOO
inline void func() {}
template <typename T>
T templ_func(T t) { return t; }
struct S {};
template <class C>
struct T { C c; };
inline int v = 43;
#endif

// If we copy foo.h directly, there are other warnings.
//--- redef.h
#ifndef REDEF
#define REDEF
inline void func() {}
template <typename T>
T templ_func(T t) { return t; }
struct S {};
template <class C>
struct T { C c; };
inline int v = 43;
#endif

//--- normal.cpp
#include "foo.h"
#include "redef.h"

// expected-error@* {{redefinition of 'func'}}
// expected-error@* {{redefinition of 'templ_func'}}
// expected-error@* {{redefinition of 'S'}}
// expected-error@* {{redefinition of 'T'}}
// expected-error@* {{redefinition of 'v'}}
// expected-note@* 1+{{previous definition is here}}

//--- M1.cppm
// These declarations are in the same TU. The compiler should complain.
module;
#include "foo.h"
#include "redef.h"
export module M1;

// expected-error@* {{redefinition of 'func'}}
// expected-error@* {{redefinition of 'templ_func'}}
// expected-error@* {{redefinition of 'S'}}
// expected-error@* {{redefinition of 'T'}}
// expected-error@* {{redefinition of 'v'}}
// expected-note@* 1+{{previous definition is here}}

//--- M2.cppm
// These declarations are in the same TU and the redefinitions are in the named modules.
// The compiler should complain.
module;
#include "foo.h"
export module M2;
#include "redef.h"

// FIXME: The diagnostic message looks not so good.
//
// expected-error@* {{declaration of 'func' in module M2 follows declaration in the global module}}
// expected-error@* {{declaration of 'templ_func' in module M2 follows declaration in the global module}}
// expected-error@* {{redefinition of 'S'}}
// expected-error@* {{redefinition of 'T'}}
// expected-error@* {{declaration of 'v' in module M2 follows declaration in the global module}}
// expected-note@* 1+{{previous definition is here}}
// expected-note@* 1+{{previous declaration is here}}

//--- M3.cppm
// These declarations are in the same TU. The compiler should complain.
export module M3;
#include "foo.h"
#include "redef.h"

// expected-error@* {{redefinition of 'func'}}
// expected-error@* {{redefinition of 'templ_func'}}
// expected-error@* {{redefinition of 'S'}}
// expected-error@* {{redefinition of 'T'}}
// expected-error@* {{redefinition of 'v'}}
// expected-note@* 1+{{previous definition is here}}

//--- M.cppm
module;
#include "foo.h"
export module M;
export using ::func;
export using ::templ_func;
export using ::S;
export using ::T;
export using ::v;

//--- Use1.cpp
// These declarations are not in the same TU. The compiler shouldn't complain.
// expected-no-diagnostics
#include "foo.h"
import M;

//--- Use2.cpp
// These declarations are not in the same TU. The compiler shouldn't complain.
// expected-no-diagnostics
import M;
#include "foo.h"

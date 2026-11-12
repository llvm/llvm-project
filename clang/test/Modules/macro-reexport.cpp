// RUN: rm -rf %t
// RUN: %clang_cc1 -fsyntax-only -DC1 -I%S/Inputs/macro-reexport %s -fmodules-cache-path=%t -verify
// RUN: %clang_cc1 -fsyntax-only -DC1 -I%S/Inputs/macro-reexport -fmodules -fimplicit-module-maps %s -fmodules-cache-path=%t -verify
// RUN: %clang_cc1 -fsyntax-only -DD1 -I%S/Inputs/macro-reexport %s -fmodules-cache-path=%t -verify
// RUN: %clang_cc1 -fsyntax-only -DD1 -I%S/Inputs/macro-reexport -fmodules -fimplicit-module-maps %s -fmodules-cache-path=%t -verify
// RUN: %clang_cc1 -fsyntax-only -DD2 -I%S/Inputs/macro-reexport %s -fmodules-cache-path=%t -verify
// RUN: %clang_cc1 -fsyntax-only -DD2 -I%S/Inputs/macro-reexport -fmodules -fimplicit-module-maps %s -fmodules-cache-path=%t -verify
// RUN: %clang_cc1 -fsyntax-only -DF1 -I%S/Inputs/macro-reexport %s -fmodules-cache-path=%t -verify
// RUN: %clang_cc1 -fsyntax-only -DF1 -I%S/Inputs/macro-reexport -fmodules -fimplicit-module-maps %s -fmodules-cache-path=%t -verify
//
// RUN: %clang_cc1 -fmodules-local-submodule-visibility -fsyntax-only -DC1 -I%S/Inputs/macro-reexport %s -fmodules-cache-path=%t -verify
// RUN: %clang_cc1 -fmodules-local-submodule-visibility -fsyntax-only -DC1 -I%S/Inputs/macro-reexport -fmodules -fimplicit-module-maps %s -fmodules-cache-path=%t -verify
// RUN: %clang_cc1 -fmodules-local-submodule-visibility -fsyntax-only -DD1 -I%S/Inputs/macro-reexport %s -fmodules-cache-path=%t -verify
// RUN: %clang_cc1 -fmodules-local-submodule-visibility -fsyntax-only -DD1 -I%S/Inputs/macro-reexport -fmodules -fimplicit-module-maps %s -fmodules-cache-path=%t -verify
// RUN: %clang_cc1 -fmodules-local-submodule-visibility -fsyntax-only -DD2 -I%S/Inputs/macro-reexport %s -fmodules-cache-path=%t -verify
// RUN: %clang_cc1 -fmodules-local-submodule-visibility -fsyntax-only -DD2 -I%S/Inputs/macro-reexport -fmodules -fimplicit-module-maps %s -fmodules-cache-path=%t -verify
// RUN: %clang_cc1 -fmodules-local-submodule-visibility -fsyntax-only -DF1 -I%S/Inputs/macro-reexport %s -fmodules-cache-path=%t -verify
// RUN: %clang_cc1 -fmodules-local-submodule-visibility -fsyntax-only -DF1 -I%S/Inputs/macro-reexport -fmodules -fimplicit-module-maps %s -fmodules-cache-path=%t -verify

#if defined(F1)
#include "f1.h"
void f() { return assert(true); } // expected-error {{undeclared identifier 'd'}}
#include "e2.h" // undefines d1's macro
void g() { return assert(true); } // expected-error {{undeclared identifier 'assert'}}
#elif defined(D1)
#include "e1.h" // undefines c1's macro but not d1's macro
#include "d1.h"
void f() { return assert(true); } // expected-error {{undeclared identifier 'd'}}
#include "e2.h" // undefines d1's macro
void g() { return assert(true); } // expected-error {{undeclared identifier 'assert'}}
#elif defined(D2)
#include "d2.h"
void f() { return assert(true); } // expected-error {{undeclared identifier 'b'}}
#else
// e2 undefines d1's macro, which overrides c1's macro.
#include "e2.h"
#include "c1.h"
void f() { return assert(true); } // expected-error {{undeclared identifier 'assert'}}
#endif

// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/submodule-visibility -verify %s -DALLOW_NAME_LEAKAGE
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-local-submodule-visibility -fmodules-cache-path=%t -I%S/Inputs/submodule-visibility -verify %s -DIMPORT
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-local-submodule-visibility -fmodules-cache-path=%t -fmodule-name=x -I%S/Inputs/submodule-visibility -verify %s
// RUN: %clang_cc1 -fimplicit-module-maps -fmodules-local-submodule-visibility -fmodules-cache-path=%t -I%S/Inputs/submodule-visibility -verify %s

#include "a.h"
#include "b.h"

#if ALLOW_NAME_LEAKAGE
// expected-no-diagnostics
#elif IMPORT
// expected-error@-6 {{could not build module 'x'}}
#else
// The use of -fmodule-name=x causes us to textually include the above headers.
// The submodule visibility rules are still applied in this case.
//
// expected-error@b.h:1 {{declaration of 'n' must be imported from module 'x.a'}}
// expected-note@a.h:1 {{here}}
#endif

int k = n + m; // OK, a and b are visible here.

#ifndef A
#error A is not defined
#endif

#ifndef B
#error B is not defined
#endif

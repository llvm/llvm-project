// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/submodule-visibility -verify %s -DALLOW_NAME_LEAKAGE
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-local-submodule-visibility -fmodules-cache-path=%t -I%S/Inputs/submodule-visibility -verify %s -DIMPORT
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-local-submodule-visibility -fmodules-cache-path=%t -fmodule-name=x -I%S/Inputs/submodule-visibility -verify %s
// RUN: %clang_cc1 -fimplicit-module-maps -fmodules-local-submodule-visibility -fmodules-cache-path=%t -I%S/Inputs/submodule-visibility -verify %s
//
// Explicit module builds.
// RUN: %clang_cc1 -fmodules -fmodules-local-submodule-visibility -emit-module -x c++-module-map %S/Inputs/submodule-visibility/module.modulemap -fmodule-name=other -o %t/other.pcm
// RUN: %clang_cc1 -fmodules -fmodule-map-file=%S/Inputs/submodule-visibility/module.modulemap -fmodules-local-submodule-visibility -fmodule-file=%t/other.pcm -verify -fmodule-name=x -I%S/Inputs/submodule-visibility %s
// RUN: %clang_cc1 -fmodules -fmodule-map-file=%S/Inputs/submodule-visibility/module.modulemap -fmodule-file=%t/other.pcm -verify -fmodule-name=x -I%S/Inputs/submodule-visibility %s -DALLOW_TEXTUAL_NAME_LEAKAGE

#include "a.h"
#include "b.h"

#if ALLOW_NAME_LEAKAGE
// expected-no-diagnostics
#elif IMPORT
// expected-error@-6 {{could not build module 'x'}}
#elif ALLOW_TEXTUAL_NAME_LEAKAGE
// expected-warning@b.h:7 {{A is defined}}
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

// Ensure we don't compute the linkage of this struct before we find it has a
// typedef name for linkage purposes.
typedef struct {
  int p;                 
  void (*f)(int p);                                                                       
} name_for_linkage;

void g() { b_template<int>(); }

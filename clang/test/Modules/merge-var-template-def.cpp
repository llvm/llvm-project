// RUN: rm -rf %t
// RUN: %clang_cc1 -I%S/Inputs/merge-var-template-def -verify -fmodules -Werror=undefined-internal -fmodules-local-submodule-visibility -fmodules-cache-path=%t -fimplicit-module-maps %s
// expected-no-diagnostics

#include "b2.h"
namespace { struct X; }
void *x = get<X>();

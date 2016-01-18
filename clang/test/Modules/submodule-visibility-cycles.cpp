// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fno-modules-error-recovery -fmodules-local-submodule-visibility -fmodules-cache-path=%t -I%S/Inputs/submodule-visibility -verify %s

#include "cycle1.h"
C1 c1;
C2 c2; // expected-error {{must be imported}} expected-error {{}}
// expected-note@cycle2.h:6 {{here}}

#include "cycle2.h"
C2 c3;

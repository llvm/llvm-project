// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/StdDef %s -verify -fno-modules-error-recovery

#include "ptrdiff_t.h"

ptrdiff_t pdt;

size_t st; // expected-error {{missing '#include "include_again.h"'; 'size_t' must be declared before it is used}}
// expected-note@stddef.h:* {{here}}

#include "include_again.h"

size_t st2;

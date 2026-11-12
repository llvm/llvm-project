// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I %S/Inputs %s -verify

struct S1 {  // expected-note{{'struct S1' begins here}}
#include "dummy.h"  // expected-error{{import of module 'dummy' appears within 'struct S1'}}
}

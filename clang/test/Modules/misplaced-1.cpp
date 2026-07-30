// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I %S/Inputs %s -verify

namespace N1 {  // expected-note{{namespace 'N1' begins here}}
#include "dummy.h"  // expected-error{{import of module 'dummy' appears within namespace 'N1'}}
}

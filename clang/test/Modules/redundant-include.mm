// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules-cache-path=%t -fmodules -fimplicit-module-maps -I%S/Inputs %s -verify
// RUN: %clang_cc1 -fmodules-cache-path=%t -fimplicit-module-maps -I%S/Inputs %s -verify

#include "cxx-header.h"
void includeNotAtTopLevel() { // expected-note {{function 'includeNotAtTopLevel' begins here}}
  #include "cxx-header.h" // expected-error {{redundant #include of module 'cxx_library' appears within function 'includeNotAtTopLevel'}}
}

namespace NS { // expected-note {{begins here}}
  #include "cxx-header.h" // expected-error {{redundant #include of module 'cxx_library' appears within namespace 'NS'}}
}

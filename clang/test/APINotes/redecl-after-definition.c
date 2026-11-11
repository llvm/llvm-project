// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fapinotes-modules -fsyntax-only -I %S/Inputs/Headers %s -verify

#include "RedeclDefinition.h"
#include "RedeclAnnotation.h"

void test(void) {
  redeclaredAfterDefinition(1); // expected-error{{'redeclaredAfterDefinition' is unavailable: not available}}
  // expected-note@Inputs/Headers/RedeclAnnotation.h:3{{'redeclaredAfterDefinition' has been explicitly marked unavailable here}}
}

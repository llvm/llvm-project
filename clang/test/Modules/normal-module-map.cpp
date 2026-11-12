// RUN: rm -rf %t
// RUN: %clang_cc1 -x objective-c -fmodules-cache-path=%t -fmodules -fimplicit-module-maps -I %S/Inputs/normal-module-map %s -verify
#include "Umbrella/umbrella_sub.h"

int getUmbrella(void) { 
  return umbrella + umbrella_sub; 
}

@import Umbrella2;

#include "a1.h"
#include "b1.h"
#include "nested/nested2.h"

int test(void) {
  return a1 + b1 + nested2;
}

@import nested_umbrella.a;

int testNestedUmbrellaA(void) {
  return nested_umbrella_a;
}

int testNestedUmbrellaBFail(void) {
  return nested_umbrella_b;
  // expected-error@-1{{declaration of 'nested_umbrella_b' must be imported from module 'nested_umbrella.b' before it is required}}
  // expected-note@Inputs/normal-module-map/nested_umbrella/b.h:1{{here}}
}

@import nested_umbrella.b;

int testNestedUmbrellaB(void) {
  return nested_umbrella_b;
}

@import nested_umbrella.a_extras;

@import nested_umbrella._1;

@import nested_umbrella.decltype_;

int testSanitizedName(void) {
  return extra_a + one + decltype_val;
}

// Note: inside the module. expected-note{{'nested_umbrella_a' declared here}}

// RUN: rm -rf %t
// RUN: %clang_cc1 -x objective-c -fmodule-cache-path %t -fmodules -I %S/Inputs/normal-module-map %s -verify
#include "Umbrella/umbrella_sub.h"

int getUmbrella() { 
  return umbrella + umbrella_sub; 
}

@__experimental_modules_import Umbrella2;

#include "a1.h"
#include "b1.h"
#include "nested/nested2.h"

int test() {
  return a1 + b1 + nested2;
}

@__experimental_modules_import nested_umbrella.a;

int testNestedUmbrellaA() {
  return nested_umbrella_a;
}

int testNestedUmbrellaBFail() {
  return nested_umbrella_b; // expected-error{{use of undeclared identifier 'nested_umbrella_b'; did you mean 'nested_umbrella_a'?}}
}

@__experimental_modules_import nested_umbrella.b;

int testNestedUmbrellaB() {
  return nested_umbrella_b;
}

@__experimental_modules_import nested_umbrella.a_extras;

@__experimental_modules_import nested_umbrella._1;

@__experimental_modules_import nested_umbrella.decltype_;

int testSanitizedName() {
  return extra_a + one + decltype_val;
}

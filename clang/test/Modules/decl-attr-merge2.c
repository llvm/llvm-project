// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t/mcache -triple arm64-apple-macosx10.7.0 \
// RUN:   -I%t/headers -fsyntax-only %t/test.c -verify

// Check more cases of attribute merging across multiple modules.

//--- headers/module.modulemap
module First {
  header "first.h" export *
}
module Second {
  header "second.h" export *
}
//--- headers/first.h
void additiveAttr(void) __attribute__((availability(macos,unavailable)));
void exclusiveAttr(void) __attribute__((hot));

//--- headers/second.h
void additiveAttr(void) __attribute__((availability(ios,introduced=4.0)));
void exclusiveAttr(void) __attribute__((cold));

//--- test.c
#include <first.h>
#include <second.h>

void test(void) {
  // Check the attribute from "second.h" doesn't hide the attribute from "first.h".
  additiveAttr();
  // expected-error@-1 {{'additiveAttr' is unavailable: not available on macOS}}
  // expected-note@first.h:* {{'additiveAttr' has been explicitly marked unavailable here}}

  // Check calling a function with `MutualExclusions` attributes.
  exclusiveAttr();
  // expected-error@second.h:* {{'cold' and 'hot' attributes are not compatible}}
  // expected-note@first.h:* {{conflicting attribute is here}}
}

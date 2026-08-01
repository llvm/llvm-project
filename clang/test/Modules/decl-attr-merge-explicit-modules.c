// Check merging attributes when modules are built explicitly.
//
// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: %clang_cc1 -fmodules -fno-implicit-modules -triple arm64-apple-macosx10.7.0 -I%t/headers \
// RUN:   -fmodule-name=first -xc %t/headers/first.modulemap -emit-module -o %t/first.pcm
// RUN: %clang_cc1 -fmodules -fno-implicit-modules -triple arm64-apple-macosx10.7.0 -I%t/headers \
// RUN:   -fmodule-name=second -xc %t/headers/second.modulemap -emit-module -o %t/second.pcm

// Without module names.
// RUN: %clang_cc1 -fmodules -fno-implicit-modules -triple arm64-apple-macosx10.7.0 -I%t/headers \
// RUN:   -fmodule-file=%t/first.pcm -fmodule-file=%t/second.pcm -fsyntax-only %t/test.c -verify
// With module names.
// RUN: %clang_cc1 -fmodules -fno-implicit-modules -triple arm64-apple-macosx10.7.0 -I%t/headers \
// RUN:   -fmodule-file=first=%t/first.pcm -fmodule-file=second=%t/second.pcm -fsyntax-only %t/test.c -verify

// Reverse order.
// RUN: %clang_cc1 -fmodules -fno-implicit-modules -triple arm64-apple-macosx10.7.0 -I%t/headers \
// RUN:   -fmodule-file=%t/second.pcm -fmodule-file=%t/first.pcm -fsyntax-only %t/test-reverse.c -verify
// RUN: %clang_cc1 -fmodules -fno-implicit-modules -triple arm64-apple-macosx10.7.0 -I%t/headers \
// RUN:   -fmodule-file=second=%t/second.pcm -fmodule-file=first=%t/first.pcm -fsyntax-only %t/test-reverse.c -verify

// With a transitive module dependency.
// RUN: %clang_cc1 -fmodules -fno-implicit-modules -triple arm64-apple-macosx10.7.0 -I%t/headers \
// RUN:   -fmodule-file=first=%t/first.pcm \
// RUN:   -fmodule-name=second_transitive -xc %t/headers/second-transitive.modulemap -emit-module -o %t/second-transitive.pcm
// RUN: %clang_cc1 -fmodules -fno-implicit-modules -triple arm64-apple-macosx10.7.0 -I%t/headers \
// RUN:   -fmodule-file=%t/second-transitive.pcm -fsyntax-only %t/test-transitive.c -verify
// RUN: %clang_cc1 -fmodules -fno-implicit-modules -triple arm64-apple-macosx10.7.0 -I%t/headers \
// RUN:   -fmodule-file=second_transitive=%t/second-transitive.pcm -fsyntax-only %t/test-transitive.c -verify

//--- headers/first.h
// Added "used" attribute to add corresponding `FunctionDecl` to `EagerlyDeserializedDecls`.
void availabilityAttr(void) __attribute__((used)) __attribute__((availability(macos,unavailable)));
//--- headers/first.modulemap
module first {
  header "first.h" export *
}

//--- headers/second.h
void availabilityAttr(void) __attribute__((used)) __attribute__((availability(ios,introduced=4.0)));
//--- headers/second.modulemap
module second {
  header "second.h" export *
}

//--- headers/second-transitive.h
#include <first.h>
void availabilityAttr(void) __attribute__((availability(ios,introduced=4.0)));
//--- headers/second-transitive.modulemap
module second_transitive {
  header "second-transitive.h" export *
}

//--- test.c
#include <first.h>
#include <second.h>
void test(void) {
  availabilityAttr();
  // expected-error@-1 {{'availabilityAttr' is unavailable: not available on macOS}}
  // expected-note@first.h:* {{'availabilityAttr' has been explicitly marked unavailable here}}
}

//--- test-reverse.c
#include <second.h>
#include <first.h>
void test(void) {
  availabilityAttr();
  // expected-error@-1 {{'availabilityAttr' is unavailable: not available on macOS}}
  // expected-note@first.h:* {{'availabilityAttr' has been explicitly marked unavailable here}}
}

//--- test-transitive.c
#include <second-transitive.h>
void test(void) {
  availabilityAttr();
  // expected-error@-1 {{'availabilityAttr' is unavailable: not available on macOS}}
  // expected-note@first.h:* {{'availabilityAttr' has been explicitly marked unavailable here}}
}

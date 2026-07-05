// RUN: rm -rf %t
// RUN: %clang_cc1 -x objective-c -fmodules-cache-path=%t -fmodules -I %S/Inputs/private %s -verify
@import libPrivate1;  // expected-error {{not found}}

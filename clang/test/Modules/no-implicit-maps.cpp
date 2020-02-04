// RUN: rm -rf %t
// RUN: %clang_cc1 -x objective-c -fmodules-cache-path=%t -fmodules -I %S/Inputs/private %s -verify
// RUN: %clang_cc1 -x objective-c -fmodules-cache-path=%t -fmodules \
// RUN:   -I %S/Inputs/private %s -verify -fimplicit-module-maps \
// RUN:   -fno-implicit-module-maps
@import libPrivate1;  // expected-error {{not found}}

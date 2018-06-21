// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules-cache-path=%t -fmodules -fimplicit-module-maps -I %S/Inputs/swift_name %s -verify
// REQUIRES: shell

@import SwiftNameInferred; // ok
@import SwiftName; // expected-error{{module 'SwiftName' not found}}

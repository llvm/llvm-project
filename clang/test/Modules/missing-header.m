// RUN: rm -rf %t
// RUN: not %clang_cc1 -x objective-c -fmodules-cache-path=%t -fmodules -fimplicit-module-maps -I %S/Inputs/submodules %s 2>&1 | FileCheck %s

// FIXME: cannot use -verify, because the error from inside the module build has
// a different source manager than the verifier.

@import missing_unavailable_headers; // OK
@import missing_unavailable_headers.not_missing; // OK
// CHECK-NOT: missing_unavailable_headers

@import missing_headers;
// CHECK: module.modulemap:15:27: error: header 'missing.h' not found
// CHECK: could not build module 'missing_headers'

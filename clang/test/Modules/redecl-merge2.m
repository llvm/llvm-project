// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I %S/Inputs %s -verify -Wno-objc-root-class
// expected-no-diagnostics

@import redecl_merge_bottom.prefix;

DeclaredThenLoaded *dtl;


@import redeclarations_left;
@import redeclarations_right;

@interface MyObject : NSObject
@end

// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -x objective-c -fmodules-cache-path=%t -emit-module -fmodule-name=redeclarations_left %S/Inputs/module.modulemap
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -x objective-c -fmodules-cache-path=%t -emit-module -fmodule-name=redeclarations_right %S/Inputs/module.modulemap
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I %S/Inputs %s -verify
// expected-no-diagnostics


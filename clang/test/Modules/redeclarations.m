@__experimental_modules_import redeclarations_left;
@__experimental_modules_import redeclarations_right;

@interface MyObject : NSObject
@end

// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -x objective-c -fmodule-cache-path %t -emit-module -fmodule-name=redeclarations_left %S/Inputs/module.map
// RUN: %clang_cc1 -fmodules -x objective-c -fmodule-cache-path %t -emit-module -fmodule-name=redeclarations_right %S/Inputs/module.map
// RUN: %clang_cc1 -fmodules -fmodule-cache-path %t %s -verify
// expected-no-diagnostics


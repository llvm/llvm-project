// RUN: rm -rf %t
// RUN: not %clang_cc1 -fmodules-cache-path=%t -fmodules -fimplicit-module-maps -F %S/Inputs -DgetModuleVersion="epic fail" %s 2>&1 | FileCheck %s

@import Module;
@import DependsOnModule;

// CHECK: While building module 'Module' imported from
// CHECK: error: expected ';' after top level declarator
// CHECK: note: expanded from macro 'getModuleVersion'
// CHECK: fatal error: could not build module 'Module'
// CHECK: While building module 'DependsOnModule' imported from
// CHECK: fatal error: could not build module 'Module'
// CHECK-NOT: error:

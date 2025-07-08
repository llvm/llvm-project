// RUN: rm -rf %t
// RUN: not %clang_cc1 -fmodules -fmodules-cache-path=%t -fmodule-map-file=%S/Inputs/export_as_test.modulemap %s 2> %t.err
// RUN: FileCheck %s < %t.err

// CHECK: export_as_test.modulemap:7:5: error: only top-level modules can be re-exported as public
// CHECK: export_as_test.modulemap:12:15: error: a module can only be re-exported as another top-level module
// CHECK: export_as_test.modulemap:3:13: error: conflicting re-export of module 'PrivateFoo' as 'Foo' or 'Bar'
// CHECK: export_as_test.modulemap:4:13: warning: module 'PrivateFoo' already re-exported as 'Bar'

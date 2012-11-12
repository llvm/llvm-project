// Test this without pch.
// RUN: %clang_cc1 -include %S/objc_stmts.h -emit-llvm -fobjc-exceptions -o - %s
// RUN: %clang_cc1 -include %S/objc_stmts.h -ast-dump -fobjc-exceptions -o - %s | FileCheck %s

// Test with pch.
// RUN: %clang_cc1 -x objective-c -emit-pch -fobjc-exceptions -o %t %S/objc_stmts.h
// RUN: %clang_cc1 -include-pch %t -emit-llvm -fobjc-exceptions -o - %s 
// RUN: %clang_cc1 -include-pch %t -ast-dump -fobjc-exceptions -o - %s | FileCheck %s

// CHECK: catch parm = "A *a"
// CHECK: catch parm = "B *b"
// CHECK: catch all

// Test this without pch.
// RUN: %clang_cc1 -triple x86_64-apple-macosx10.10 -include %S/objc_stmts.h -emit-llvm -fobjc-exceptions -o - %s
// RUN: %clang_cc1 -triple x86_64-apple-macosx10.10 -include %S/objc_stmts.h -ast-print -fobjc-exceptions -o - %s | FileCheck %s

// Test with pch.
// RUN: %clang_cc1 -triple x86_64-apple-macosx10.10 -x objective-c -emit-pch -fobjc-exceptions -o %t %S/objc_stmts.h
// RUN: %clang_cc1 -triple x86_64-apple-macosx10.10 -include-pch %t -emit-llvm -fobjc-exceptions -o - %s 
// RUN: %clang_cc1 -triple x86_64-apple-macosx10.10 -include-pch %t -ast-print -fobjc-exceptions -o - %s | FileCheck %s

// CHECK: @catch(A *a)
// CHECK: @catch(B *b)
// CHECK: @catch()

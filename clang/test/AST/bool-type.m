// RUN: %clang_cc1 -triple thumbv7s-apple-ios8.0 -ast-dump "%s" 2>&1 | FileCheck %s --check-prefix CHECK-CHAR
// RUN: %clang_cc1 -triple thumbv7k-apple-watchos2.0 -ast-dump "%s" 2>&1 | FileCheck %s --check-prefix CHECK-BOOL
// RUN: %clang_cc1 -triple i386-apple-watchos2.0 -ast-dump "%s" 2>&1 | FileCheck %s --check-prefix CHECK-BOOL
// RUN: %clang_cc1 -triple arm64-apple-ios8.0 -ast-dump "%s" 2>&1 | FileCheck %s --check-prefix CHECK-BOOL
// RUN: %clang_cc1 -triple x86_64-apple-ios8.0 -ast-dump "%s" 2>&1 | FileCheck %s --check-prefix CHECK-BOOL
// RUN: %clang_cc1 -triple i386-apple-macosx10.10 -ast-dump "%s" 2>&1 | FileCheck %s --check-prefix CHECK-CHAR
// RUN: %clang_cc1 -triple x86_64-apple-macosx10.10 -ast-dump "%s" 2>&1 | FileCheck %s --check-prefix CHECK-CHAR

// CHECK-CHAR: ObjCBoolLiteralExpr {{.*}} 'signed char' __objc_yes
// CHECK-BOOL: ObjCBoolLiteralExpr {{.*}} '_Bool' __objc_yes

int var = __objc_yes;

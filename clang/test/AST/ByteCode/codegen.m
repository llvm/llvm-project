// REQUIRES: x86-registered-target

/// See test/CodeGenObjC/constant-strings.m
/// Test that we let the APValue we create for ObjCStringLiterals point to the right expression.

// RUN: %clang_cc1 -triple x86_64-macho -emit-llvm -o %t %s -fexperimental-new-constant-interpreter
// RUN: FileCheck --check-prefix=CHECK-NEXT < %t %s

// Check that we set alignment 1 on the string.
//
// CHECK-NEXT: @.str = {{.*}}constant [13 x i8] c"Hello World!\00", section "__TEXT,__cstring,cstring_literals", align 1
id a = @"Hello World!";

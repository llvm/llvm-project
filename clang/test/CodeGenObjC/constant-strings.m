// REQUIRES: x86-registered-target

// RUN: %clang_cc1 -triple x86_64-macho -emit-llvm -o %t %s
// RUN: FileCheck --check-prefix=CHECK-NEXT < %t %s

// Check that we set alignment 1 on the string.
//
// CHECK-NEXT: @.str = {{.*}}constant [13 x i8] c"Hello World!\00", section "__TEXT,__cstring,cstring_literals", align 1

// RUN: %clang_cc1 -triple x86_64-macho -fobjc-runtime=gcc -emit-llvm -o %t %s
// RUN: FileCheck --check-prefix=CHECK-GNU < %t %s
// CHECK-GNU: NXConstantString

// RUN: %clang_cc1 -triple x86_64-macho -fobjc-runtime=gcc -fconstant-string-class NSConstantString -emit-llvm -o %t %s
// RUN: FileCheck --check-prefix=CHECK-GNU-WITH-CLASS < %t %s
// CHECK-GNU-WITH-CLASS: NSConstantString
//
// RUN: %clang_cc1 -triple x86_64-unknown-freebsd -fobjc-runtime=gnustep-2.0 -emit-llvm -o %t %s
// RUN: FileCheck --check-prefix=CHECK-GNUSTEP2 < %t %s

// CHECK-GNUSTEP2: @._OBJC_CLASS_NSConstantString = external global i8*
// CHECK-GNUSTEP2: @0 = private unnamed_addr constant [13 x i8] c"Hello World!\00", align 1
// CHECK-GNUSTEP2: @.objc_string = private global { i8**, i32, i32, i32, i32, i8* } { i8** @._OBJC_CLASS_NSConstantString, i32 0, i32 12, i32 12, i32 0, i8* getelementptr inbounds ([13 x i8], [13 x i8]* @0, i64 0, i64 0) }, section "__objc_constant_string", align 8
// CHECK-GNUSTEP2: @b = global i8* inttoptr (i64 -3340545023602065388 to i8*), align 8
// CHECK-GNUSTEP2: @.objc_str_Hello_World = linkonce_odr hidden global { i8**, i32, i32, i32, i32, i8* } { i8** @._OBJC_CLASS_NSConstantString, i32 0, i32 11, i32 11, i32 0, i8* getelementptr inbounds ([12 x i8], [12 x i8]* @1, i64 0, i64 0) }, section "__objc_constant_string", comdat, align 8
// CHECK-GNUSTEP2: @c =
// CHECK-SAME-GNUSTEP2: @.objc_str_Hello_World
//
id a = @"Hello World!";
id b = @"hi";
id c = @"Hello World";

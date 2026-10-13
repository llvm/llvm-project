// RUN: split-file %s %t

// RUN: %clang_cc1 %t/pragma-copyright.c -triple powerpc-ibm-aix   -emit-llvm -disable-llvm-passes -o - | FileCheck %s  --check-prefix=CHECK-COPYRIGHT
// RUN: %clang_cc1 %t/pragma-copyright.c -triple powerpc64-ibm-aix -emit-llvm -disable-llvm-passes -o - | FileCheck %s  --check-prefix=CHECK-COPYRIGHT

// RUN: %clang_cc1 %t/operator-pragma-copyright.c -triple powerpc-ibm-aix   -emit-llvm -disable-llvm-passes -o - | FileCheck %s --check-prefix=CHECK-OPERATOR
// RUN: %clang_cc1 %t/operator-pragma-copyright.c -triple powerpc64-ibm-aix -emit-llvm -disable-llvm-passes -o - | FileCheck %s --check-prefix=CHECK-OPERATOR

//--- pragma-copyright.c
#pragma comment(copyright, "@(#) Hello, " " world\n\t\"quoted\"")

int main() { return 0; }

// CHECK-COPYRIGHT: @[[LOADTIME_STR:__loadtime_comment_str_[0-9a-f]+]] = weak_odr hidden unnamed_addr constant [29 x i8] c"@(#) Hello, world\0A\09\22quoted\22\00", section "__loadtime_comment", align 1, !loadtime_comment !0
// CHECK-COPYRIGHT: @llvm.compiler.used = appending global [1 x ptr] [ptr @[[LOADTIME_STR]]], section "llvm.metadata"

//--- operator-pragma-copyright.c
_Pragma("comment(copyright, \"IBM Copyright Pragma Operator\")")
void foo() {}

// CHECK-OPERATOR: @[[LOADTIME_STR:__loadtime_comment_str_[0-9a-f]+]] = weak_odr hidden unnamed_addr constant [30 x i8] c"IBM Copyright Pragma Operator\00", section "__loadtime_comment", align 1, !loadtime_comment !0
// CHECK-OPERATOR: @llvm.compiler.used = appending global [1 x ptr] [ptr @[[LOADTIME_STR]]], section "llvm.metadata"

// RUN: %clang_cc1 -O1 -triple s390x-linux-gnu -emit-llvm %s -o - | FileCheck %s

// #include <stdio.h>

// CHECK: @msg1 = local_unnamed_addr constant [13 x i8] c"Hello World\0A\00", align 2
// CHECK: @str = private unnamed_addr constant [12 x i8] c"Hello World\00", align 2

const char msg1 [] = "Hello World\n";

extern int printf(const char *__restrict __format, ...);

void foo() {
    printf(msg1);
}

// REQUIRES: systemz-registered-target
// RUN: %clang --target=s390x-none-zos -S -emit-llvm %s -o - | FileCheck %s

// Check the variables
// CHECK: @func_ptr = global ptr null, align 8
// CHECK: @var1 = global i32 0, align 4
// CHECK: @var2 = hidden global i32 0, align 4
// CHECK: @var3 = global i32 0, align 4
// CHECK: @var4 = hidden global i32 0, align 4
// CHECK: @var5 = global i32 0, align 4

// Check the functions
// CHECK: define void @foo1
// CHECK: define hidden void @foo2

int _Export var1;
int var2;
int _Export var3, var4, _Export var5;

void _Export foo1(){};
void foo2(){};

int (*_Export func_ptr)(void) = 0;

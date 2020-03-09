// RUN: %clang_cc1 -triple i686-pc-linux -emit-llvm < %s | FileCheck %s
// All of these should uses the memory representation of _Bool

// CHECK-LABEL: %struct.teststruct1 = type { i8, i8 }
// CHECK-LABEL: @test1 = common global %struct.teststruct1
struct teststruct1 {_Bool a, b;} test1;

// CHECK-LABEL: @test2 = common global i8* null
_Bool* test2;

// CHECK-LABEL: @test3 = common global [10 x i8]
_Bool test3[10];

// CHECK-LABEL: @test4 = common global [0 x i8]* null
_Bool (*test4)[];

// CHECK-LABEL: define void @f(i32 %x)
void f(int x) {
  // CHECK: alloca i8, align 1
  _Bool test5;

  // CHECK: alloca i8, i32 %{{.*}}, align 1
  _Bool test6[x];
}

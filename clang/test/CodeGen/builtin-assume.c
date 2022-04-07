// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-unknown-unknown -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -no-opaque-pointers -triple i386-mingw32 -fms-extensions -emit-llvm -o - %s | FileCheck %s

int nonconst(void);
int isconst(void) __attribute__((const));
int ispure(void) __attribute__((pure));

// CHECK-LABEL: @test1
int test1(int *a, int i) {
// CHECK: store i32* %a, i32** [[A_ADDR:%.+]], align
// CHECK: [[A:%.+]] = load i32*, i32** [[A_ADDR]]
// CHECK: [[CMP:%.+]] = icmp ne i32* [[A]], null
// CHECK: call void @llvm.assume(i1 [[CMP]])

// CHECK: [[CALL:%.+]] = call i32 @isconst()
// CHECK: [[BOOL:%.+]] = icmp ne i32 [[CALL]], 0
// CHECK: call void @llvm.assume(i1 [[BOOL]])

// CHECK: [[CALLPURE:%.+]] = call i32 @ispure()
// CHECK: [[BOOLPURE:%.+]] = icmp ne i32 [[CALLPURE]], 0
// CHECK: call void @llvm.assume(i1 [[BOOLPURE]])
#ifdef _MSC_VER
  __assume(a != 0)
  __assume(isconst());
  __assume(ispure());
#else
  __builtin_assume(a != 0);
  __builtin_assume(isconst());
  __builtin_assume(ispure());
#endif

// Nothing is generated for an assume with side effects...
// CHECK-NOT: load i32*, i32** %i.addr
// CHECK-NOT: call void @llvm.assume
// CHECK-NOT: call i32 @nonconst()
#ifdef _MSC_VER
  __assume(++i != 0)
  __assume(nonconst());
#else
  __builtin_assume(++i != 0);
  __builtin_assume(nonconst());
#endif

  return a[0];
}


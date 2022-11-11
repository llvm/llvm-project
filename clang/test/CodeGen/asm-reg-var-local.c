// RUN: %clang_cc1 %s -triple x86_64-apple-darwin -emit-llvm -o - | FileCheck %s
// Exercise various use cases for local asm "register variables".

int foo(void) {
// CHECK-LABEL: define{{.*}} i32 @foo()
// CHECK: [[A:%[a-zA-Z0-9]+]] = alloca i32

  register int a asm("rsi")=5;
// CHECK: store i32 5, ptr [[A]]

  asm volatile("; %0 This asm defines rsi" : "=r"(a));
// CHECK: [[Z:%[a-zA-Z0-9]+]] = call i32 asm sideeffect "; $0 This asm defines rsi", "={rsi},~{dirflag},~{fpsr},~{flags}"()
// CHECK: store i32 [[Z]], ptr [[A]]

  a = 42;
// CHECK:  store i32 42, ptr [[A]]

  asm volatile("; %0 This asm uses rsi" : : "r"(a));
// CHECK:  [[TMP:%[a-zA-Z0-9]+]] = load i32, ptr [[A]]
// CHECK:  call void asm sideeffect "; $0 This asm uses rsi", "{rsi},~{dirflag},~{fpsr},~{flags}"(i32 [[TMP]])

  return a;
// CHECK:  [[TMP1:%[a-zA-Z0-9]+]] = load i32, ptr [[A]]
// CHECK:  ret i32 [[TMP1]]
}

int earlyclobber(void) {
// CHECK-LABEL: define{{.*}} i32 @earlyclobber()
// CHECK: [[A:%[a-zA-Z0-9]+]] = alloca i32

  register int a asm("rsi")=5;
// CHECK: store i32 5, ptr [[A]]

  asm volatile("; %0 This asm defines rsi" : "=&r"(a));
// CHECK: [[Z:%[a-zA-Z0-9]+]] = call i32 asm sideeffect "; $0 This asm defines rsi", "=&{rsi},~{dirflag},~{fpsr},~{flags}"()
// CHECK: store i32 [[Z]], ptr [[A]]

  a = 42;
// CHECK:  store i32 42, ptr [[A]]

  asm volatile("; %0 This asm uses rsi" : : "r"(a));
// CHECK:  [[TMP:%[a-zA-Z0-9]+]] = load i32, ptr [[A]]
// CHECK:  call void asm sideeffect "; $0 This asm uses rsi", "{rsi},~{dirflag},~{fpsr},~{flags}"(i32 [[TMP]])

  return a;
// CHECK:  [[TMP1:%[a-zA-Z0-9]+]] = load i32, ptr [[A]]
// CHECK:  ret i32 [[TMP1]]
}

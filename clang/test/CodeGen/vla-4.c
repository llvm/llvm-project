// RUN: %clang_cc1 -no-opaque-pointers -triple i386-unknown-unknown %s -emit-llvm -o - | FileCheck %s

int f(void);
int h(void);

void t1(void) {
  _Atomic(typeof((int (*)[f()]) h())) v;
  // CHECK:      [[N:%.*]] = alloca i32*, align 4
  // CHECK-NEXT: [[P:%.*]] = call i32 @f
  // CHECK-NEXT: [[P:%.*]] = call i32 @h
}

void t2(void) {
  typeof(typeof((int (*)[f()]) h())) v;
  // CHECK:      [[N:%.*]] = alloca i32*, align 4
  // CHECK-NEXT: [[P:%.*]] = call i32 @f
  // CHECK-NEXT: [[P:%.*]] = call i32 @h
}

void t3(typeof((int (*)[f()]) h()) v) {
  // CHECK:      store i32* %v, i32** %{{[.0-9A-Za-z]+}}, align 4
  // CHECK-NEXT: [[P:%.*]] = call i32 @f
  // CHECK-NEXT: [[P:%.*]] = call i32 @h
}

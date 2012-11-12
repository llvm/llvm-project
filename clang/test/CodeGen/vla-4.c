// RUN: %clang_cc1 -triple i386-unknown-unknown %s -emit-llvm -o - | FileCheck %s

int f();
int h();

void t1() {
  _Atomic(typeof((int (*)[f()]) h())) v;
  // CHECK:      [[N:%.*]] = alloca i32*, align 4
  // CHECK-NEXT: [[P:%.*]] = call i32 bitcast (i32 (...)* @f to i32 ()*)()
  // CHECK-NEXT: [[P:%.*]] = call i32 bitcast (i32 (...)* @h to i32 ()*)()
}

void t2() {
  typeof(typeof((int (*)[f()]) h())) v;
  // CHECK:      [[N:%.*]] = alloca i32*, align 4
  // CHECK-NEXT: [[P:%.*]] = call i32 bitcast (i32 (...)* @f to i32 ()*)()
  // CHECK-NEXT: [[P:%.*]] = call i32 bitcast (i32 (...)* @h to i32 ()*)()
}

void t3(typeof((int (*)[f()]) h()) v) {
  // CHECK:      store i32* %v, i32** %{{[.0-9A-Za-z]+}}, align 4
  // CHECK-NEXT: [[P:%.*]] = call i32 bitcast (i32 (...)* @f to i32 ()*)()
  // CHECK-NEXT: [[P:%.*]] = call i32 bitcast (i32 (...)* @h to i32 ()*)()
}

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -std=c++17 %s -o - | FileCheck %s

void bar(const int &i = 42);

void foo() {
  bar();
}

// CHECK: [[TMP0:%.*]] = cir.alloca !s32i
// CHECK: [[TMP1:%.*]] = cir.const #cir.int<42>
// CHECK: cir.store{{.*}} [[TMP1]], [[TMP0]]
// CHECK: cir.call @_Z3barRKi([[TMP0]])

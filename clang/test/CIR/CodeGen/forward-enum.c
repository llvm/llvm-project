// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck %s --input-file=%t.cir

extern enum X x;
void f(void) {
  x;
}

enum X {
  One,
  Two
};

// CHECK: cir.global "private" external @x : !u32i
// CHECK: cir.func{{.*}} @f
// CHECK:   cir.get_global @x : !cir.ptr<!u32i>

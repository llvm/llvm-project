// RUN: %clang -cc1 -triple x86_64-linux-pc -O3 -disable-llvm-passes %s -emit-llvm -o - | FileCheck %s

void Shifts(_BitInt(12) E, int i) {
  E << 99;
  // CHECK: shl i12 %{{.+}}, 3

  77 << E;
  // CHECK: %[[PROM:.+]] = zext i12 %{{.+}} to i32
  // CHECK: %[[MASK:.+]] = and i32 %[[PROM]], 31
  // CHECK: shl i32 77, %[[MASK]]

  E << i;
  // CHECK: %[[PROM:.+]] = trunc i32 %{{.+}} to i12
  // CHECK: %[[MASK:.+]] = urem i12 %[[PROM]], 12
  // CHECK: shl i12 %{{.+}}, %[[MASK]]

  i << E;
  // CHECK: %[[PROM:.+]] = zext i12 %{{.+}} to i32
  // CHECK: %[[MASK:.+]] = and i32 %[[PROM]], 31
  // CHECK: shl i32 %{{.+}}, %[[MASK]]
}

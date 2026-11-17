// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c23 -ffixed-point -emit-llvm -o - %s | FileCheck %s

// Conversion of a fixed-point value to an enumeration with a bool underlying
// type has the semantics of conversion to bool.

enum Bool : bool { BF, BT };

// CHECK-LABEL: define {{.*}} i32 @from_fixed_point(
// CHECK:         [[CMP:%.*]] = icmp ne i32 {{%.*}}, 0
int from_fixed_point(_Accum x) {
  return (enum Bool)x;
}

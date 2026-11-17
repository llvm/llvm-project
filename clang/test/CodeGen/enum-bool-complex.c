// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c23 -emit-llvm -o - %s | FileCheck %s

// Conversion of a complex value to an enumeration with a bool underlying type
// has the semantics of conversion to bool: the result is true when either the
// real or the imaginary part is non-zero.

enum Bool : bool { BF, BT };

// CHECK-LABEL: define {{.*}} i32 @from_complex(
// CHECK:         [[RE:%.*]] = fcmp une double {{%.*}}, 0.000000e+00
// CHECK:         [[IM:%.*]] = fcmp une double {{%.*}}, 0.000000e+00
// CHECK:         [[OR:%.*]] = or i1 [[RE]], [[IM]]
int from_complex(double _Complex x) {
  return (enum Bool)x;
}

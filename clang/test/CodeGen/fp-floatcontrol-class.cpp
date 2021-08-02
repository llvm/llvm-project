// RUN: %clang_cc1 -ffp-contract=on -triple x86_64-linux-gnu -emit-llvm -o - %s | FileCheck %s
// Verify that float_control does not pertain to initializer expressions

float y();
float z();
#pragma float_control(except, on)
class ON {
  float w = 2 + y() * z();
  // CHECK-LABEL: define {{.*}} @_ZN2ONC2Ev{{.*}}
  // CHECK: llvm.experimental.constrained.fmul{{.*}}tonearest{{.*}}strict
};
ON on;
#pragma float_control(except, off)
class OFF {
  float w = 2 + y() * z();
  // CHECK-LABEL: define {{.*}} @_ZN3OFFC2Ev{{.*}}
  // CHECK-NOT: llvm.experimental.constrained.fmul{{.*}}tonearest{{.*}}strict
};
OFF off;

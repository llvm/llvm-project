// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck --check-prefix=CHECK-DEF %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -ffp-exception-behavior=strict -emit-llvm -o - %s | FileCheck --check-prefix=CHECK-STRICT %s

// REQUIRES: x86-registered-target

float func_01(float x, float y, float z) {
  float res = x + y;
  {
#pragma clang fp exceptions(maytrap)
    res += z;
  }
  return res;
}
// CHECK-DEF-LABEL: @_Z7func_01fff
// CHECK-DEF:       call float @llvm.experimental.constrained.fadd.f32(float {{.*}}, float {{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// CHECK-DEF:       call float @llvm.experimental.constrained.fadd.f32(float {{.*}}, float {{.*}}, metadata !"round.tonearest", metadata !"fpexcept.maytrap")

// CHECK-STRICT-LABEL: @_Z7func_01fff
// CHECK-STRICT:       call float @llvm.experimental.constrained.fadd.f32(float {{.*}}, float {{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK-STRICT:       call float @llvm.experimental.constrained.fadd.f32(float {{.*}}, float {{.*}}, metadata !"round.tonearest", metadata !"fpexcept.maytrap")

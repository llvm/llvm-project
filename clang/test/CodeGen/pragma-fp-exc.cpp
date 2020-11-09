// RUN: %clang_cc1 -triple %itanium_abi_triple -emit-llvm -o - %s | FileCheck --check-prefix=CHECK-DEF %s
// RUN: %clang_cc1 -triple %itanium_abi_triple -ffp-exception-behavior=strict -emit-llvm -o - %s | FileCheck --check-prefix=CHECK-STRICT %s

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

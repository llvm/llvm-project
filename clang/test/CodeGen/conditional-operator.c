// RUN: %clang_cc1 -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -disable-llvm-passes -emit-llvm -menable-no-infs -fapprox-func\
// RUN: -funsafe-math-optimizations -fno-signed-zeros -mreassociate -freciprocal-math\
// RUN: -ffp-contract=fast -ffast-math %s -o - | FileCheck %s

float test_precise_off_select(int c) {
#pragma float_control(precise, off)
  return c ? 0.0f : 1.0f;
}

// CHECK-LABEL: test_precise_off_select
// CHECK: select fast i1 {{%.+}}, float 0.000000e+00, float 1.000000e+00

float test_precise_off_phi(int c, float t, float f) {
#pragma float_control(precise, off)
    return c ? t : f;
}

// CHECK-LABEL: test_precise_off_phi
// CHECK: phi fast float [ {{%.+}}, {{%.+}} ], [ {{%.+}}, {{%.+}} ]

float test_precise_on_select(int c) {
#pragma float_control(precise, on)
    return c ? 0.0f : 1.0f;
}

// CHECK-LABEL: test_precise_on_select
// CHECK: select i1 {{%.+}}, float 0.000000e+00, float 1.000000e+00

float test_precise_on_phi(int c, float t, float f) {
#pragma float_control(precise, on)
  return c ? t : f;
}

// CHECK-LABEL: test_precise_on_phi
// CHECK: phi float [ {{%.+}}, {{%.+}} ], [ {{%.+}}, {{%.+}} ]

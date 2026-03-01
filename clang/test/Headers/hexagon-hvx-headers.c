// REQUIRES: hexagon-registered-target

// RUN: %clang_cc1 -O0 -internal-isystem %S/../../lib/Headers/ \
// RUN:   -target-cpu hexagonv68 -triple hexagon-unknown-elf \
// RUN:   -target-feature +hvx-length128b -target-feature +hvxv68 \
// RUN:   -emit-llvm %s -o - | FileCheck --check-prefix=CHECK %s

// RUN: %clang_cc1 -O0 -internal-isystem %S/../../lib/Headers/ \
// RUN:   -target-cpu hexagonv68 -triple hexagon-unknown-elf -DDIRECT \
// RUN:   -target-feature +hvx-length128b -target-feature +hvxv68 \
// RUN:   -emit-llvm %s -o - | FileCheck --check-prefix=CHECK %s

// RUN: %clang_cc1 -O0 -internal-isystem %S/../../lib/Headers/ \
// RUN:   -target-cpu hexagonv68 -triple hexagon-unknown-elf -x c++ \
// RUN:   -target-feature +hvx-length128b -target-feature +hvxv68 \
// RUN:   -emit-llvm %s -o - | FileCheck --check-prefix=CHECK %s

// RUN: %clang_cc1 -O0 -internal-isystem %S/../../lib/Headers/ \
// RUN:   -target-cpu hexagonv68 -triple hexagon-unknown-elf \
// RUN:   -target-feature +hvx-length64b -target-feature +hvxv68 \
// RUN:   -emit-llvm %s -o - | FileCheck --check-prefix=CHECK-64 %s

// Test that HVXIEEEFP intrinsics are available with +hvx-ieee-fp.
// RUN: %clang_cc1 -O0 -internal-isystem %S/../../lib/Headers/ \
// RUN:   -target-cpu hexagonv79 -triple hexagon-unknown-elf \
// RUN:   -target-feature +hvx-length128b -target-feature +hvxv79 \
// RUN:   -target-feature +hvx-ieee-fp \
// RUN:   -emit-llvm %s -o - | FileCheck --check-prefix=CHECK-IEEEFP %s

// Test that HVXIEEEFP intrinsics are NOT visible without +hvx-ieee-fp.
// RUN: %clang_cc1 -O0 -internal-isystem %S/../../lib/Headers/ \
// RUN:   -target-cpu hexagonv79 -triple hexagon-unknown-elf \
// RUN:   -target-feature +hvx-length128b -target-feature +hvxv79 \
// RUN:   -fsyntax-only %s

#ifdef DIRECT
#include <hvx_hexagon_protos.h>
#else
#include <hexagon_protos.h>
#endif
#include <hexagon_types.h>

// expected-no-diagnostics

void test_hvx_protos(float a, unsigned int b) {
  HVX_VectorPair c;
  // CHECK-64: call <32 x i32> @llvm.hexagon.V6.v6mpyhubs10
  // CHECK:    call <64 x i32> @llvm.hexagon.V6.v6mpyhubs10.128B
  c = Q6_Ww_v6mpy_WubWbI_h(c, c, 2);
}

#ifdef __HVX_IEEE_FP__
void test_hvx_ieeefp(HVX_Vector a, HVX_Vector b) {
  // CHECK-IEEEFP: call <32 x i32> @llvm.hexagon.V6.vabs.hf.128B
  HVX_Vector r0 = Q6_Vhf_vabs_Vhf(a);
#if __HVX_ARCH__ >= 73
  // CHECK-IEEEFP: call <32 x i32> @llvm.hexagon.V6.vcvt.bf.sf.128B
  HVX_Vector r1 = Q6_Vbf_vcvt_VsfVsf(a, b);
#endif
#if __HVX_ARCH__ >= 79
  // CHECK-IEEEFP: call <32 x i32> @llvm.hexagon.V6.vfneg.f8.128B
  HVX_Vector r2 = Q6_V_vfneg_V(a);
#endif
}
#else
// Verify IEEE FP intrinsic macros are not defined without +hvx-ieee-fp.
#ifdef Q6_Vhf_vabs_Vhf
#error "Q6_Vhf_vabs_Vhf should not be defined without __HVX_IEEE_FP__"
#endif
#ifdef Q6_Vbf_vcvt_VsfVsf
#error "Q6_Vbf_vcvt_VsfVsf should not be defined without __HVX_IEEE_FP__"
#endif
#ifdef Q6_V_vfneg_V
#error "Q6_V_vfneg_V should not be defined without __HVX_IEEE_FP__"
#endif
#endif

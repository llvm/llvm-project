// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -target-feature +altivec -target-feature +vsx \
// RUN:   -triple powerpc64-unknown-unknown -emit-llvm %s -o - \
// RUN:   -D__XL_COMPAT_ALTIVEC__ -target-cpu pwr7 | FileCheck %s
// RUN: %clang_cc1 -target-feature +altivec -target-feature +vsx \
// RUN:   -triple powerpc64le-unknown-unknown -emit-llvm %s -o - \
// RUN:   -D__XL_COMPAT_ALTIVEC__ -target-cpu pwr8 | FileCheck %s
#include <altivec.h>
vector double vd = { 3.4e22, 1.8e-3 };
vector signed long long vsll = { -12345678999ll, 12345678999 };
vector unsigned long long vull = { 11547229456923630743llu, 18014402265226391llu };
vector float res_vf;
vector signed int res_vsi;
vector unsigned int res_vui;

void test() {
// CHECK-LABEL: @test(
// CHECK-NEXT:  entry:
// CHECK-LE-LABEL: @test(
// CHECK-LE-NEXT:  entry:

  res_vf = vec_ctf(vsll, 4);
// CHECK:         [[TMP0:%.*]] = load <2 x i64>, <2 x i64>* @vsll, align 16
// CHECK-NEXT:    [[TMP1:%.*]] = call <4 x float> @llvm.ppc.vsx.xvcvsxdsp(<2 x i64> [[TMP0]])
// CHECK-NEXT:    fmul <4 x float> [[TMP1]], <float 6.250000e-02, float 6.250000e-02, float 6.250000e-02, float 6.250000e-02>

  res_vf = vec_ctf(vull, 4);
// CHECK:         [[TMP2:%.*]] = load <2 x i64>, <2 x i64>* @vull, align 16
// CHECK-NEXT:    [[TMP3:%.*]] = call <4 x float> @llvm.ppc.vsx.xvcvuxdsp(<2 x i64> [[TMP2]])
// CHECK-NEXT:    fmul <4 x float> [[TMP3]], <float 6.250000e-02, float 6.250000e-02, float 6.250000e-02, float 6.250000e-02>

  res_vsi = vec_cts(vd, 4);
// CHECK:         [[TMP4:%.*]] = load <2 x double>, <2 x double>* @vd, align 16
// CHECK-NEXT:    fmul <2 x double> [[TMP4]], <double 1.600000e+01, double 1.600000e+01>
// CHECK:         call <4 x i32> @llvm.ppc.vsx.xvcvdpsxws(<2 x double>

  res_vui = vec_ctu(vd, 4);
// CHECK:         [[TMP8:%.*]] = load <2 x double>, <2 x double>* @vd, align 16
// CHECK-NEXT:    fmul <2 x double> [[TMP8]], <double 1.600000e+01, double 1.600000e+01>
// CHECK:         call <4 x i32> @llvm.ppc.vsx.xvcvdpuxws(<2 x double>
}

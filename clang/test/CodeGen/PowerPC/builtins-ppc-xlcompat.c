// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -flax-vector-conversions=none -target-feature +altivec -target-feature +vsx \
// RUN:   -triple powerpc64-unknown-linux-gnu -emit-llvm %s -o - \
// RUN:   -D__XL_COMPAT_ALTIVEC__ -target-cpu pwr7 | FileCheck %s
// RUN: %clang_cc1 -flax-vector-conversions=none -target-feature +altivec -target-feature +vsx \
// RUN:   -triple powerpc64le-unknown-linux-gnu -emit-llvm %s -o - \
// RUN:   -D__XL_COMPAT_ALTIVEC__ -target-cpu pwr8 | FileCheck %s
// RUN: %clang_cc1 -flax-vector-conversions=none -target-feature +altivec -target-feature +vsx \
// RUN:   -triple powerpc64le-unknown-linux-gnu -emit-llvm %s -o - \
// RUN:   -U__XL_COMPAT_ALTIVEC__ -target-cpu pwr8 | FileCheck \
// RUN:   --check-prefix=NOCOMPAT %s
#include <altivec.h>
vector double vd = { 3.4e22, 1.8e-3 };
vector signed long long res_vsll, vsll = { -12345678999ll, 12345678999 };
vector unsigned long long res_vull, vull = { 11547229456923630743llu, 18014402265226391llu };
vector float res_vf;
vector double res_vd;
vector signed int res_vsi;
vector unsigned int res_vui;

void test() {
// CHECK-LABEL: @test(
// CHECK-NEXT:  entry:
// NOCOMPAT-LABEL: @test(
// NOCOMPAT-NEXT:  entry:

  res_vf = vec_ctf(vsll, 4);
// CHECK:         [[TMP0:%.*]] = load <2 x i64>, ptr @vsll, align 16
// CHECK-NEXT:    [[TMP1:%.*]] = call <4 x float> @llvm.ppc.vsx.xvcvsxdsp(<2 x i64> [[TMP0]])
// CHECK-NEXT:    fmul <4 x float> [[TMP1]], splat (float 6.250000e-02)
// NOCOMPAT:      [[TMP0:%.*]] = load <2 x i64>, ptr @vsll, align 16
// NOCOMPAT-NEXT: [[CONV:%.*]] = sitofp <2 x i64> [[TMP0]] to <2 x double>
// NOCOMPAT-NEXT: fmul <2 x double> [[CONV]], splat (double 6.250000e-02)

  res_vf = vec_ctf(vull, 4);
// CHECK:         [[TMP2:%.*]] = load <2 x i64>, ptr @vull, align 16
// CHECK-NEXT:    [[TMP3:%.*]] = call <4 x float> @llvm.ppc.vsx.xvcvuxdsp(<2 x i64> [[TMP2]])
// CHECK-NEXT:    fmul <4 x float> [[TMP3]], splat (float 6.250000e-02)
// NOCOMPAT:      [[TMP2:%.*]] = load <2 x i64>, ptr @vull, align 16
// NOCOMPAT-NEXT: [[CONV1:%.*]] = uitofp <2 x i64> [[TMP2]] to <2 x double>
// NOCOMPAT-NEXT: fmul <2 x double> [[CONV1]], splat (double 6.250000e-02)

  res_vsll = vec_cts(vd, 4);
// CHECK:         [[TMP4:%.*]] = load <2 x double>, ptr @vd, align 16
// CHECK-NEXT:    fmul <2 x double> [[TMP4]], splat (double 1.600000e+01)
// CHECK:         call <4 x i32> @llvm.ppc.vsx.xvcvdpsxws(<2 x double>
// NOCOMPAT:      [[TMP4:%.*]] = load <2 x double>, ptr @vd, align 16
// NOCOMPAT-NEXT: fmul <2 x double> [[TMP4]], splat (double 1.600000e+01)

  res_vull = vec_ctu(vd, 4);
// CHECK:         [[TMP8:%.*]] = load <2 x double>, ptr @vd, align 16
// CHECK-NEXT:    fmul <2 x double> [[TMP8]], splat (double 1.600000e+01)
// CHECK:         call <4 x i32> @llvm.ppc.vsx.xvcvdpuxws(<2 x double>
// NOCOMPAT:      [[TMP7:%.*]] = load <2 x double>, ptr @vd, align 16
// NOCOMPAT-NEXT: fmul <2 x double> [[TMP7]], splat (double 1.600000e+01)

  res_vd = vec_round(vd);
// CHECK:         call double @llvm.ppc.readflm()
// CHECK:         call double @llvm.ppc.setrnd(i32 0)
// CHECK:         call <2 x double> @llvm.rint.v2f64(<2 x double>
// CHECK:         call double @llvm.ppc.setflm(double
// NOCOMPAT:      call <2 x double> @llvm.round.v2f64(<2 x double>
}

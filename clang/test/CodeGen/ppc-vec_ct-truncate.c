// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -target-feature +altivec -target-feature +vsx \
// RUN:   -flax-vector-conversions=all -triple powerpc64-ibm-aix-xcoff -emit-llvm %s -o - \
// RUN:   -target-cpu pwr7 | FileCheck %s
// RUN: %clang_cc1 -target-feature +altivec -target-feature +vsx \
// RUN:   -flax-vector-conversions=all -triple powerpc64-unknown-linux-gnu -emit-llvm %s -o - \
// RUN:   -target-cpu pwr8 | FileCheck %s
// RUN: %clang_cc1 -target-feature +altivec -target-feature +vsx \
// RUN:   -flax-vector-conversions=all -triple powerpc64le-unknown-linux-gnu -emit-llvm %s -o - \
// RUN:   -target-cpu pwr8 | FileCheck %s
// RUN: %clang_cc1 -target-feature +altivec -target-feature +vsx \
// RUN:   -flax-vector-conversions=all -triple powerpc64-ibm-aix-xcoff -emit-llvm %s -o - \
// RUN:   -D__XL_COMPAT_ALTIVEC__ -target-cpu pwr7 | FileCheck %s
// RUN: %clang_cc1 -target-feature +altivec -target-feature +vsx \
// RUN:   -flax-vector-conversions=all -triple powerpc64le-unknown-linux-gnu -emit-llvm %s -o - \
// RUN:   -D__XL_COMPAT_ALTIVEC__ -target-cpu pwr8 | FileCheck %s

#include <altivec.h>
vector double a1 = {-1.234e-5, 1.2345};
vector signed int res_vsi;
vector float vf1 = {0.234, 1.234, 2.345, 3.456};
vector signed int vsi1 = {1, 2, 3, 4};
vector double res_vd;
vector float res_vf;
vector signed long long res_vsll;
vector unsigned long long res_vull;
void test(void) {
  // CHECK-LABEL: @test(
  // CHECK-NEXT:  entry:

  res_vsi = vec_cts(a1, 31);
  //  CHECK:       [[TMP0:%.*]] = load <2 x double>, ptr @a1, align 16
  //  CHECK-NEXT:  fmul <2 x double> [[TMP0]], splat (double 0x41E0000000000000)

  res_vsi = vec_cts(a1, 500);
  // CHECK:        [[TMP4:%.*]] = load <2 x double>, ptr @a1, align 16
  // CHECK-NEXT:   fmul <2 x double> [[TMP4]], splat (double 0x4130000000000000)

  res_vsi = vec_ctu(vf1, 31);
  // CHECK:        [[TMP8:%.*]] = load <4 x float>, ptr @vf1, align 16
  // CHECK-NEXT:   call <4 x i32> @llvm.ppc.altivec.vctuxs(<4 x float> [[TMP8]], i32 31)

  res_vsi = vec_ctu(vf1, 500);
  // CHECK:        [[TMP10:%.*]] = load <4 x float>, ptr @vf1, align 16
  // CHECK-NEXT:   call <4 x i32> @llvm.ppc.altivec.vctuxs(<4 x float> [[TMP10]], i32 20)

  res_vull = vec_ctul(vf1, 31);
  // CHECK:        [[TMP12:%.*]] = load <4 x float>, ptr @vf1, align 16
  // CHECK-NEXT:   fmul <4 x float> [[TMP12]], splat (float 0x41E0000000000000)

  res_vull = vec_ctul(vf1, 500);
  // CHECK:        [[TMP21:%.*]] = load <4 x float>, ptr @vf1, align 16
  // CHECK-NEXT:   fmul <4 x float> [[TMP21]], splat (float 0x4130000000000000)

  res_vsll = vec_ctsl(vf1, 31);
  // CHECK:        [[TMP30:%.*]] = load <4 x float>, ptr @vf1, align 16
  // CHECK-NEXT:   fmul <4 x float> [[TMP30]], splat (float 0x41E0000000000000)

  res_vsll = vec_ctsl(vf1, 500);
  // CHECK:        [[TMP39:%.*]] = load <4 x float>, ptr @vf1, align 16
  // CHECK-NEXT:   fmul <4 x float> [[TMP39]], splat (float 0x4130000000000000)

  res_vf = vec_ctf(vsi1, 31);
  // CHECK:        [[TMP48:%.*]] = load <4 x i32>, ptr @vsi1, align 16
  // CHECK-NEXT:   call <4 x float> @llvm.ppc.altivec.vcfsx(<4 x i32> [[TMP48]], i32 31)

  res_vf = vec_ctf(vsi1, 500);
  // CHECK:        [[TMP50:%.*]] = load <4 x i32>, ptr @vsi1, align 16
  // CHECK-NEXT:   call <4 x float> @llvm.ppc.altivec.vcfsx(<4 x i32> [[TMP50]], i32 20)

  res_vd = vec_ctd(vsi1, 31);
  // CHECK:        [[TMP53:%.*]] = load <4 x i32>, ptr @vsi1, align 16
  // CHECK:        [[TMP83:%.*]] = call <2 x double> @llvm.ppc.vsx.xvcvsxwdp(<4 x i32> [[TMP82:%.*]])
  // CHECK-NEXT:   fmul <2 x double> [[TMP83]], splat (double 0x3E00000000000000)

  res_vd = vec_ctd(vsi1, 500);
  // CHECK:        [[TMP84:%.*]] = load <4 x i32>, ptr @vsi1, align 16
  // CHECK:        [[TMP115:%.*]] = call <2 x double> @llvm.ppc.vsx.xvcvsxwdp(<4 x i32> [[TMP114:%.*]])
  // CHECK-NEXT:   fmul <2 x double> [[TMP115]], splat (double 0x3EB0000000000000)
}

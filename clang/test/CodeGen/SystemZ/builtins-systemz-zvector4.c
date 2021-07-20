// REQUIRES: systemz-registered-target
// RUN: %clang_cc1 -target-cpu arch14 -triple s390x-linux-gnu \
// RUN: -O2 -fzvector -flax-vector-conversions=none \
// RUN: -Wall -Wno-unused -Werror -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -target-cpu arch14 -triple s390x-linux-gnu \
// RUN: -O2 -fzvector -flax-vector-conversions=none \
// RUN: -Wall -Wno-unused -Werror -S %s -o - | FileCheck %s --check-prefix=CHECK-ASM

#include <vecintrin.h>

volatile vector signed char vsc;
volatile vector signed short vss;
volatile vector signed int vsi;
volatile vector signed long long vsl;
volatile vector unsigned char vuc;
volatile vector unsigned short vus;
volatile vector unsigned int vui;
volatile vector unsigned long long vul;
volatile vector bool char vbc;
volatile vector bool short vbs;
volatile vector bool int vbi;
volatile vector bool long long vbl;
volatile vector float vf;
volatile vector double vd;

void test_nnp_assist(void) {
  // CHECK-ASM-LABEL: test_nnp_assist

  vf = vec_extend_to_fp32_hi(vus, 0);
  // CHECK: call <4 x float> @llvm.s390.vclfnhs(<8 x i16> %{{.*}}, i32 0)
  // CHECK-ASM: vclfnh
  vf = vec_extend_to_fp32_hi(vus, 15);
  // CHECK: call <4 x float> @llvm.s390.vclfnhs(<8 x i16> %{{.*}}, i32 15)
  // CHECK-ASM: vclfnh

  vf = vec_extend_to_fp32_lo(vus, 0);
  // CHECK: call <4 x float> @llvm.s390.vclfnls(<8 x i16> %{{.*}}, i32 0)
  // CHECK-ASM: vclfnl
  vf = vec_extend_to_fp32_lo(vus, 15);
  // CHECK: call <4 x float> @llvm.s390.vclfnls(<8 x i16> %{{.*}}, i32 15)
  // CHECK-ASM: vclfnl

  vus = vec_round_from_fp32(vf, vf, 0);
  // CHECK: call <8 x i16> @llvm.s390.vcrnfs(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 0)
  // CHECK-ASM: vcrnf
  vus = vec_round_from_fp32(vf, vf, 15);
  // CHECK: call <8 x i16> @llvm.s390.vcrnfs(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 15)
  // CHECK-ASM: vcrnf

  vus = vec_convert_to_fp16(vus, 0);
  // CHECK: call <8 x i16> @llvm.s390.vcfn(<8 x i16> %{{.*}} i32 0)
  // CHECK-ASM: vcfn
  vus = vec_convert_to_fp16(vus, 15);
  // CHECK: call <8 x i16> @llvm.s390.vcfn(<8 x i16> %{{.*}} i32 15)
  // CHECK-ASM: vcfn

  vus = vec_convert_from_fp16(vus, 0);
  // CHECK: call <8 x i16> @llvm.s390.vcnf(<8 x i16> %{{.*}} i32 0)
  // CHECK-ASM: vcnf
  vus = vec_convert_from_fp16(vus, 15);
  // CHECK: call <8 x i16> @llvm.s390.vcnf(<8 x i16> %{{.*}} i32 15)
  // CHECK-ASM: vcnf
}

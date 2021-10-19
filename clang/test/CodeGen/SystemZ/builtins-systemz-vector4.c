// REQUIRES: systemz-registered-target
// RUN: %clang_cc1 -target-cpu arch14 -triple s390x-ibm-linux -flax-vector-conversions=none \
// RUN: -Wall -Wno-unused -Werror -emit-llvm %s -o - | FileCheck %s

typedef __attribute__((vector_size(16))) signed char vec_schar;
typedef __attribute__((vector_size(16))) signed short vec_sshort;
typedef __attribute__((vector_size(16))) signed int vec_sint;
typedef __attribute__((vector_size(16))) signed long long vec_slong;
typedef __attribute__((vector_size(16))) unsigned char vec_uchar;
typedef __attribute__((vector_size(16))) unsigned short vec_ushort;
typedef __attribute__((vector_size(16))) unsigned int vec_uint;
typedef __attribute__((vector_size(16))) unsigned long long vec_ulong;
typedef __attribute__((vector_size(16))) double vec_double;
typedef __attribute__((vector_size(16))) float vec_float;

volatile vec_schar vsc;
volatile vec_sshort vss;
volatile vec_sint vsi;
volatile vec_slong vsl;
volatile vec_uchar vuc;
volatile vec_ushort vus;
volatile vec_uint vui;
volatile vec_ulong vul;
volatile vec_double vd;
volatile vec_float vf;

void test_nnp_assist(void) {
  vf = __builtin_s390_vclfnhs(vus, 0);
  // CHECK: call <4 x float> @llvm.s390.vclfnhs(<8 x i16> %{{.*}}, i32 0)
  vf = __builtin_s390_vclfnhs(vus, 15);
  // CHECK: call <4 x float> @llvm.s390.vclfnhs(<8 x i16> %{{.*}}, i32 15)

  vf = __builtin_s390_vclfnls(vus, 0);
  // CHECK: call <4 x float> @llvm.s390.vclfnls(<8 x i16> %{{.*}}, i32 0)
  vf = __builtin_s390_vclfnls(vus, 15);
  // CHECK: call <4 x float> @llvm.s390.vclfnls(<8 x i16> %{{.*}}, i32 15)

  vus = __builtin_s390_vcrnfs(vf, vf, 0);
  // CHECK: call <8 x i16> @llvm.s390.vcrnfs(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 0)
  vus = __builtin_s390_vcrnfs(vf, vf, 15);
  // CHECK: call <8 x i16> @llvm.s390.vcrnfs(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 15)

  vus = __builtin_s390_vcfn(vus, 0);
  // CHECK: call <8 x i16> @llvm.s390.vcfn(<8 x i16> %{{.*}} i32 0)
  vus = __builtin_s390_vcfn(vus, 15);
  // CHECK: call <8 x i16> @llvm.s390.vcfn(<8 x i16> %{{.*}} i32 15)

  vus = __builtin_s390_vcnf(vus, 0);
  // CHECK: call <8 x i16> @llvm.s390.vcnf(<8 x i16> %{{.*}} i32 0)
  vus = __builtin_s390_vcnf(vus, 15);
  // CHECK: call <8 x i16> @llvm.s390.vcnf(<8 x i16> %{{.*}} i32 15)
}


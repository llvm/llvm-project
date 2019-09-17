// REQUIRES: systemz-registered-target
// RUN: %clang_cc1 -target-cpu z14 -triple s390x-ibm-linux -flax-vector-conversions=none \
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

volatile unsigned int len;
const void * volatile cptr;
void * volatile ptr;
int cc;

void test_core(void) {
  vul = __builtin_s390_vbperm(vuc, vuc);
  // CHECK: call <2 x i64> @llvm.s390.vbperm(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})

  vsc = __builtin_s390_vlrl(len, cptr);
  // CHECK: call <16 x i8> @llvm.s390.vlrl(i32 %{{.*}}, i8* %{{.*}})

  __builtin_s390_vstrl(vsc, len, ptr);
  // CHECK: call void @llvm.s390.vstrl(<16 x i8> %{{.*}}, i32 %{{.*}}, i8* %{{.*}})
}

void test_integer(void) {
  vuc = __builtin_s390_vmslg(vul, vul, vuc, 0);
  // CHECK: call <16 x i8> @llvm.s390.vmslg(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  vuc = __builtin_s390_vmslg(vul, vul, vuc, 15);
  // CHECK: call <16 x i8> @llvm.s390.vmslg(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <16 x i8> %{{.*}}, i32 15)
}

void test_float(void) {
  vd = __builtin_s390_vfmaxdb(vd, vd, 4);
  // CHECK: call <2 x double> @llvm.maxnum.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}})
  vd = __builtin_s390_vfmaxdb(vd, vd, 0);
  // CHECK: call <2 x double> @llvm.s390.vfmaxdb(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 0)
  vd = __builtin_s390_vfmaxdb(vd, vd, 15);
  // CHECK: call <2 x double> @llvm.s390.vfmaxdb(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 15)

  vd = __builtin_s390_vfmindb(vd, vd, 4);
  // CHECK: call <2 x double> @llvm.minnum.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}})
  vd = __builtin_s390_vfmindb(vd, vd, 0);
  // CHECK: call <2 x double> @llvm.s390.vfmindb(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 0)
  vd = __builtin_s390_vfmindb(vd, vd, 15);
  // CHECK: call <2 x double> @llvm.s390.vfmindb(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 15)

  vd = __builtin_s390_vfnmadb(vd, vd, vd);
  // CHECK: [[RES:%[^ ]+]] = call <2 x double> @llvm.fma.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}})
  // CHECK: fsub <2 x double> <double -0.000000e+00, double -0.000000e+00>, [[RES]]
  vd = __builtin_s390_vfnmsdb(vd, vd, vd);
  // CHECK: [[NEG:%[^ ]+]] = fsub <2 x double> <double -0.000000e+00, double -0.000000e+00>, %{{.*}}
  // CHECK: [[RES:%[^ ]+]] = call <2 x double> @llvm.fma.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> [[NEG]])
  // CHECK: fsub <2 x double> <double -0.000000e+00, double -0.000000e+00>, [[RES]]

  vsi = __builtin_s390_vfcesbs(vf, vf, &cc);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vfcesbs(<4 x float> %{{.*}}, <4 x float> %{{.*}})
  vsi = __builtin_s390_vfchsbs(vf, vf, &cc);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vfchsbs(<4 x float> %{{.*}}, <4 x float> %{{.*}})
  vsi = __builtin_s390_vfchesbs(vf, vf, &cc);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vfchesbs(<4 x float> %{{.*}}, <4 x float> %{{.*}})

  vsi = __builtin_s390_vftcisb(vf, 0, &cc);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vftcisb(<4 x float> %{{.*}}, i32 0)
  vsi = __builtin_s390_vftcisb(vf, 4095, &cc);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vftcisb(<4 x float> %{{.*}}, i32 4095)

  vf = __builtin_s390_vfmaxsb(vf, vf, 4);
  // CHECK: call <4 x float> @llvm.maxnum.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}})
  vf = __builtin_s390_vfmaxsb(vf, vf, 0);
  // CHECK: call <4 x float> @llvm.s390.vfmaxsb(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 0)
  vf = __builtin_s390_vfmaxsb(vf, vf, 15);
  // CHECK: call <4 x float> @llvm.s390.vfmaxsb(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 15)

  vf = __builtin_s390_vfminsb(vf, vf, 4);
  // CHECK: call <4 x float> @llvm.minnum.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}})
  vf = __builtin_s390_vfminsb(vf, vf, 0);
  // CHECK: call <4 x float> @llvm.s390.vfminsb(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 0)
  vf = __builtin_s390_vfminsb(vf, vf, 15);
  // CHECK: call <4 x float> @llvm.s390.vfminsb(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 15)

  vf = __builtin_s390_vfsqsb(vf);
  // CHECK: call <4 x float> @llvm.sqrt.v4f32(<4 x float> %{{.*}})

  vf = __builtin_s390_vfmasb(vf, vf, vf);
  // CHECK: call <4 x float> @llvm.fma.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}})
  vf = __builtin_s390_vfmssb(vf, vf, vf);
  // CHECK: [[NEG:%[^ ]+]] = fsub <4 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %{{.*}}
  // CHECK: call <4 x float> @llvm.fma.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> [[NEG]])
  vf = __builtin_s390_vfnmasb(vf, vf, vf);
  // CHECK: [[RES:%[^ ]+]] = call <4 x float> @llvm.fma.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}})
  // CHECK: fsub <4 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, [[RES]]
  vf = __builtin_s390_vfnmssb(vf, vf, vf);
  // CHECK: [[NEG:%[^ ]+]] = fsub <4 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %{{.*}}
  // CHECK: [[RES:%[^ ]+]] = call <4 x float> @llvm.fma.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> [[NEG]])
  // CHECK: fsub <4 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, [[RES]]

  vf = __builtin_s390_vflpsb(vf);
  // CHECK: call <4 x float> @llvm.fabs.v4f32(<4 x float> %{{.*}})
  vf = __builtin_s390_vflnsb(vf);
  // CHECK: [[ABS:%[^ ]+]] = call <4 x float> @llvm.fabs.v4f32(<4 x float> %{{.*}})
  // CHECK: fsub <4 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, [[ABS]]

  vf = __builtin_s390_vfisb(vf, 0, 0);
  // CHECK: call <4 x float> @llvm.rint.v4f32(<4 x float> %{{.*}})
  vf = __builtin_s390_vfisb(vf, 4, 0);
  // CHECK: call <4 x float> @llvm.nearbyint.v4f32(<4 x float> %{{.*}})
  vf = __builtin_s390_vfisb(vf, 4, 1);
  // CHECK: call <4 x float> @llvm.round.v4f32(<4 x float> %{{.*}})
  vf = __builtin_s390_vfisb(vf, 4, 5);
  // CHECK: call <4 x float> @llvm.trunc.v4f32(<4 x float> %{{.*}})
  vf = __builtin_s390_vfisb(vf, 4, 6);
  // CHECK: call <4 x float> @llvm.ceil.v4f32(<4 x float> %{{.*}})
  vf = __builtin_s390_vfisb(vf, 4, 7);
  // CHECK: call <4 x float> @llvm.floor.v4f32(<4 x float> %{{.*}})
  vf = __builtin_s390_vfisb(vf, 4, 4);
  // CHECK: call <4 x float> @llvm.s390.vfisb(<4 x float> %{{.*}}, i32 4, i32 4)
}


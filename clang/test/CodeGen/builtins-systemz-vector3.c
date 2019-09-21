// REQUIRES: systemz-registered-target
// RUN: %clang_cc1 -target-cpu z15 -triple s390x-ibm-linux -flax-vector-conversions=none \
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

void test_integer(void) {
  vuc = __builtin_s390_vsld(vuc, vuc, 0);
  // CHECK: call <16 x i8> @llvm.s390.vsld(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  vuc = __builtin_s390_vsld(vuc, vuc, 7);
  // CHECK: call <16 x i8> @llvm.s390.vsld(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 7)

  vuc = __builtin_s390_vsrd(vuc, vuc, 0);
  // CHECK: call <16 x i8> @llvm.s390.vsrd(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  vuc = __builtin_s390_vsrd(vuc, vuc, 7);
  // CHECK: call <16 x i8> @llvm.s390.vsrd(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 7)
}

void test_string(void) {
  vuc = __builtin_s390_vstrsb(vuc, vuc, vuc, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vstrsb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  vuc = __builtin_s390_vstrsh(vus, vus, vuc, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vstrsh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <16 x i8> %{{.*}})
  vuc = __builtin_s390_vstrsf(vui, vui, vuc, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vstrsf(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <16 x i8> %{{.*}})

  vuc = __builtin_s390_vstrszb(vuc, vuc, vuc, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vstrszb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  vuc = __builtin_s390_vstrszh(vus, vus, vuc, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vstrszh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <16 x i8> %{{.*}})
  vuc = __builtin_s390_vstrszf(vui, vui, vuc, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vstrszf(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <16 x i8> %{{.*}})
}


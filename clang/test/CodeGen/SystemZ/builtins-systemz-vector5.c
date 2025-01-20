// REQUIRES: systemz-registered-target
// RUN: %clang_cc1 -target-cpu arch15 -triple s390x-ibm-linux -flax-vector-conversions=none \
// RUN: -Wall -Wno-unused -Werror -emit-llvm %s -o - | FileCheck %s

typedef __attribute__((vector_size(16))) signed char vec_schar;
typedef __attribute__((vector_size(16))) signed short vec_sshort;
typedef __attribute__((vector_size(16))) signed int vec_sint;
typedef __attribute__((vector_size(16))) signed long long vec_slong;
typedef __attribute__((vector_size(16))) signed __int128 vec_sint128;
typedef __attribute__((vector_size(16))) unsigned char vec_uchar;
typedef __attribute__((vector_size(16))) unsigned short vec_ushort;
typedef __attribute__((vector_size(16))) unsigned int vec_uint;
typedef __attribute__((vector_size(16))) unsigned long long vec_ulong;
typedef __attribute__((vector_size(16))) unsigned __int128 vec_uint128;
typedef __attribute__((vector_size(16))) double vec_double;

volatile vec_schar vsc;
volatile vec_sshort vss;
volatile vec_sint vsi;
volatile vec_slong vsl;
volatile vec_uchar vuc;
volatile vec_ushort vus;
volatile vec_uint vui;
volatile vec_ulong vul;
volatile signed __int128 si128;
volatile unsigned __int128 ui128;

int cc;

void test_core(void) {
  vuc = __builtin_s390_vgemb(vus);
  // CHECK: call <16 x i8> @llvm.s390.vgemb(<8 x i16> %{{.*}})
  vus = __builtin_s390_vgemh(vuc);
  // CHECK: call <8 x i16> @llvm.s390.vgemh(<16 x i8> %{{.*}})
  vui = __builtin_s390_vgemf(vuc);
  // CHECK: call <4 x i32> @llvm.s390.vgemf(<16 x i8> %{{.*}})
  vul = __builtin_s390_vgemg(vuc);
  // CHECK: call <2 x i64> @llvm.s390.vgemg(<16 x i8> %{{.*}})
  ui128 = __builtin_s390_vgemq(vuc);
  // CHECK: call i128 @llvm.s390.vgemq(<16 x i8> %{{.*}})

  si128 = __builtin_s390_vuphg(vsl);
  // CHECK: call i128 @llvm.s390.vuphg(<2 x i64> %{{.*}})
  si128 = __builtin_s390_vuplg(vsl);
  // CHECK: call i128 @llvm.s390.vuplg(<2 x i64> %{{.*}})
  ui128 = __builtin_s390_vuplhg(vul);
  // CHECK: call i128 @llvm.s390.vuplhg(<2 x i64> %{{.*}})
  ui128 = __builtin_s390_vupllg(vul);
  // CHECK: call i128 @llvm.s390.vupllg(<2 x i64> %{{.*}})
}

void test_integer(void) {
  si128 = __builtin_s390_vavgq(si128, si128);
  // CHECK: call i128 @llvm.s390.vavgq(i128 %{{.*}}, i128 %{{.*}})
  ui128 = __builtin_s390_vavglq(ui128, ui128);
  // CHECK: call i128 @llvm.s390.vavglq(i128 %{{.*}}, i128 %{{.*}})

  vuc = __builtin_s390_veval(vuc, vuc, vuc, 0);
  // CHECK: call <16 x i8> @llvm.s390.veval(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)

  vsl = __builtin_s390_vmahg(vsl, vsl, vsl);
  // CHECK: call <2 x i64> @llvm.s390.vmahg(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  si128 = __builtin_s390_vmahq(si128, si128, si128);
  // CHECK: call i128 @llvm.s390.vmahq(i128 %{{.*}}, i128 %{{.*}}, i128 %{{.*}})
  vul = __builtin_s390_vmalhg(vul, vul, vul);
  // CHECK: call <2 x i64> @llvm.s390.vmalhg(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  ui128 = __builtin_s390_vmalhq(ui128, ui128, ui128);
  // CHECK: call i128 @llvm.s390.vmalhq(i128 %{{.*}}, i128 %{{.*}}, i128 %{{.*}})

  si128 = __builtin_s390_vmaeg(vsl, vsl, si128);
  // CHECK: call i128 @llvm.s390.vmaeg(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, i128 %{{.*}})
  ui128 = __builtin_s390_vmaleg(vul, vul, ui128);
  // CHECK: call i128 @llvm.s390.vmaleg(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, i128 %{{.*}})
  si128 = __builtin_s390_vmaog(vsl, vsl, si128);
  // CHECK: call i128 @llvm.s390.vmaog(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, i128 %{{.*}})
  ui128 = __builtin_s390_vmalog(vul, vul, ui128);
  // CHECK: call i128 @llvm.s390.vmalog(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, i128 %{{.*}})

  vsl = __builtin_s390_vmhg(vsl, vsl);
  // CHECK: call <2 x i64> @llvm.s390.vmhg(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  si128 = __builtin_s390_vmhq(si128, si128);
  // CHECK: call i128 @llvm.s390.vmhq(i128 %{{.*}}, i128 %{{.*}})
  vul = __builtin_s390_vmlhg(vul, vul);
  // CHECK: call <2 x i64> @llvm.s390.vmlhg(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  ui128 = __builtin_s390_vmlhq(ui128, ui128);
  // CHECK: call i128 @llvm.s390.vmlhq(i128 %{{.*}}, i128 %{{.*}})

  si128 = __builtin_s390_vmeg(vsl, vsl);
  // CHECK: call i128 @llvm.s390.vmeg(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  ui128 = __builtin_s390_vmleg(vul, vul);
  // CHECK: call i128 @llvm.s390.vmleg(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  si128 = __builtin_s390_vmog(vsl, vsl);
  // CHECK: call i128 @llvm.s390.vmog(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  ui128 = __builtin_s390_vmlog(vul, vul);
  // CHECK: call i128 @llvm.s390.vmlog(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})

  si128 = __builtin_s390_vceqqs(ui128, ui128, &cc);
  // CHECK: call { i128, i32 } @llvm.s390.vceqqs(i128 %{{.*}}, i128 %{{.*}})
  si128 = __builtin_s390_vchqs(si128, si128, &cc);
  // CHECK: call { i128, i32 } @llvm.s390.vchqs(i128 %{{.*}}, i128 %{{.*}})
  si128 = __builtin_s390_vchlqs(ui128, ui128, &cc);
  // CHECK: call { i128, i32 } @llvm.s390.vchlqs(i128 %{{.*}}, i128 %{{.*}})
}

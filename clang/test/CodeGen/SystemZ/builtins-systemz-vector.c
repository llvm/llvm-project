// REQUIRES: systemz-registered-target
// RUN: %clang_cc1 -no-opaque-pointers -target-cpu z13 -triple s390x-ibm-linux -flax-vector-conversions=none \
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

volatile vec_schar vsc;
volatile vec_sshort vss;
volatile vec_sint vsi;
volatile vec_slong vsl;
volatile vec_uchar vuc;
volatile vec_ushort vus;
volatile vec_uint vui;
volatile vec_ulong vul;
volatile vec_double vd;

volatile unsigned int len;
const void * volatile cptr;
void * volatile ptr;
int cc;

void test_core(void) {
  len = __builtin_s390_lcbb(cptr, 0);
  // CHECK: call i32 @llvm.s390.lcbb(i8* %{{.*}}, i32 0)
  len = __builtin_s390_lcbb(cptr, 15);
  // CHECK: call i32 @llvm.s390.lcbb(i8* %{{.*}}, i32 15)

  vsc = __builtin_s390_vlbb(cptr, 0);
  // CHECK: call <16 x i8> @llvm.s390.vlbb(i8* %{{.*}}, i32 0)
  vsc = __builtin_s390_vlbb(cptr, 15);
  // CHECK: call <16 x i8> @llvm.s390.vlbb(i8* %{{.*}}, i32 15)

  vsc = __builtin_s390_vll(len, cptr);
  // CHECK: call <16 x i8> @llvm.s390.vll(i32 %{{.*}}, i8* %{{.*}})

  vul = __builtin_s390_vpdi(vul, vul, 0);
  // CHECK: call <2 x i64> @llvm.s390.vpdi(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, i32 0)
  vul = __builtin_s390_vpdi(vul, vul, 15);
  // CHECK: call <2 x i64> @llvm.s390.vpdi(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, i32 15)

  vuc = __builtin_s390_vperm(vuc, vuc, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vperm(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}})

  vuc = __builtin_s390_vpklsh(vus, vus);
  // CHECK: call <16 x i8> @llvm.s390.vpklsh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  vus = __builtin_s390_vpklsf(vui, vui);
  // CHECK: call <8 x i16> @llvm.s390.vpklsf(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  vui = __builtin_s390_vpklsg(vul, vul);
  // CHECK: call <4 x i32> @llvm.s390.vpklsg(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})

  vuc = __builtin_s390_vpklshs(vus, vus, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vpklshs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  vus = __builtin_s390_vpklsfs(vui, vui, &cc);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vpklsfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  vui = __builtin_s390_vpklsgs(vul, vul, &cc);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vpklsgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})

  vsc = __builtin_s390_vpksh(vss, vss);
  // CHECK: call <16 x i8> @llvm.s390.vpksh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  vss = __builtin_s390_vpksf(vsi, vsi);
  // CHECK: call <8 x i16> @llvm.s390.vpksf(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  vsi = __builtin_s390_vpksg(vsl, vsl);
  // CHECK: call <4 x i32> @llvm.s390.vpksg(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})

  vsc = __builtin_s390_vpkshs(vss, vss, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vpkshs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  vss = __builtin_s390_vpksfs(vsi, vsi, &cc);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vpksfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  vsi = __builtin_s390_vpksgs(vsl, vsl, &cc);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vpksgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})

  __builtin_s390_vstl(vsc, len, ptr);
  // CHECK: call void @llvm.s390.vstl(<16 x i8> %{{.*}}, i32 %{{.*}}, i8* %{{.*}})

  vss = __builtin_s390_vuphb(vsc);
  // CHECK: call <8 x i16> @llvm.s390.vuphb(<16 x i8> %{{.*}})
  vsi = __builtin_s390_vuphh(vss);
  // CHECK: call <4 x i32> @llvm.s390.vuphh(<8 x i16> %{{.*}})
  vsl = __builtin_s390_vuphf(vsi);
  // CHECK: call <2 x i64> @llvm.s390.vuphf(<4 x i32> %{{.*}})

  vss = __builtin_s390_vuplb(vsc);
  // CHECK: call <8 x i16> @llvm.s390.vuplb(<16 x i8> %{{.*}})
  vsi = __builtin_s390_vuplhw(vss);
  // CHECK: call <4 x i32> @llvm.s390.vuplhw(<8 x i16> %{{.*}})
  vsl = __builtin_s390_vuplf(vsi);
  // CHECK: call <2 x i64> @llvm.s390.vuplf(<4 x i32> %{{.*}})

  vus = __builtin_s390_vuplhb(vuc);
  // CHECK: call <8 x i16> @llvm.s390.vuplhb(<16 x i8> %{{.*}})
  vui = __builtin_s390_vuplhh(vus);
  // CHECK: call <4 x i32> @llvm.s390.vuplhh(<8 x i16> %{{.*}})
  vul = __builtin_s390_vuplhf(vui);
  // CHECK: call <2 x i64> @llvm.s390.vuplhf(<4 x i32> %{{.*}})

  vus = __builtin_s390_vupllb(vuc);
  // CHECK: call <8 x i16> @llvm.s390.vupllb(<16 x i8> %{{.*}})
  vui = __builtin_s390_vupllh(vus);
  // CHECK: call <4 x i32> @llvm.s390.vupllh(<8 x i16> %{{.*}})
  vul = __builtin_s390_vupllf(vui);
  // CHECK: call <2 x i64> @llvm.s390.vupllf(<4 x i32> %{{.*}})
}

void test_integer(void) {
  vuc = __builtin_s390_vaq(vuc, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vaq(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  vuc = __builtin_s390_vacq(vuc, vuc, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vacq(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  vuc = __builtin_s390_vaccq(vuc, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vaccq(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  vuc = __builtin_s390_vacccq(vuc, vuc, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vacccq(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}})

  vuc = __builtin_s390_vaccb(vuc, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vaccb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  vus = __builtin_s390_vacch(vus, vus);
  // CHECK: call <8 x i16> @llvm.s390.vacch(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  vui = __builtin_s390_vaccf(vui, vui);
  // CHECK: call <4 x i32> @llvm.s390.vaccf(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  vul = __builtin_s390_vaccg(vul, vul);
  // CHECK: call <2 x i64> @llvm.s390.vaccg(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})

  vsc = __builtin_s390_vavgb(vsc, vsc);
  // CHECK: call <16 x i8> @llvm.s390.vavgb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  vss = __builtin_s390_vavgh(vss, vss);
  // CHECK: call <8 x i16> @llvm.s390.vavgh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  vsi = __builtin_s390_vavgf(vsi, vsi);
  // CHECK: call <4 x i32> @llvm.s390.vavgf(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  vsl = __builtin_s390_vavgg(vsl, vsl);
  // CHECK: call <2 x i64> @llvm.s390.vavgg(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})

  vuc = __builtin_s390_vavglb(vuc, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vavglb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  vus = __builtin_s390_vavglh(vus, vus);
  // CHECK: call <8 x i16> @llvm.s390.vavglh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  vui = __builtin_s390_vavglf(vui, vui);
  // CHECK: call <4 x i32> @llvm.s390.vavglf(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  vul = __builtin_s390_vavglg(vul, vul);
  // CHECK: call <2 x i64> @llvm.s390.vavglg(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})

  vui = __builtin_s390_vcksm(vui, vui);
  // CHECK: call <4 x i32> @llvm.s390.vcksm(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})

  vuc = __builtin_s390_vclzb(vuc);
  // CHECK: call <16 x i8> @llvm.ctlz.v16i8(<16 x i8> %{{.*}}, i1 false)
  vus = __builtin_s390_vclzh(vus);
  // CHECK: call <8 x i16> @llvm.ctlz.v8i16(<8 x i16> %{{.*}}, i1 false)
  vui = __builtin_s390_vclzf(vui);
  // CHECK: call <4 x i32> @llvm.ctlz.v4i32(<4 x i32> %{{.*}}, i1 false)
  vul = __builtin_s390_vclzg(vul);
  // CHECK: call <2 x i64> @llvm.ctlz.v2i64(<2 x i64> %{{.*}}, i1 false)

  vuc = __builtin_s390_vctzb(vuc);
  // CHECK: call <16 x i8> @llvm.cttz.v16i8(<16 x i8> %{{.*}}, i1 false)
  vus = __builtin_s390_vctzh(vus);
  // CHECK: call <8 x i16> @llvm.cttz.v8i16(<8 x i16> %{{.*}}, i1 false)
  vui = __builtin_s390_vctzf(vui);
  // CHECK: call <4 x i32> @llvm.cttz.v4i32(<4 x i32> %{{.*}}, i1 false)
  vul = __builtin_s390_vctzg(vul);
  // CHECK: call <2 x i64> @llvm.cttz.v2i64(<2 x i64> %{{.*}}, i1 false)

  vuc = __builtin_s390_verimb(vuc, vuc, vuc, 0);
  // CHECK: call <16 x i8> @llvm.s390.verimb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  vuc = __builtin_s390_verimb(vuc, vuc, vuc, 255);
  // CHECK: call <16 x i8> @llvm.s390.verimb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 255)
  vus = __builtin_s390_verimh(vus, vus, vus, 0);
  // CHECK: call <8 x i16> @llvm.s390.verimh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i32 0)
  vus = __builtin_s390_verimh(vus, vus, vus, 255);
  // CHECK: call <8 x i16> @llvm.s390.verimh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i32 255)
  vui = __builtin_s390_verimf(vui, vui, vui, 0);
  // CHECK: call <4 x i32> @llvm.s390.verimf(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 0)
  vui = __builtin_s390_verimf(vui, vui, vui, 255);
  // CHECK: call <4 x i32> @llvm.s390.verimf(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 255)
  vul = __builtin_s390_verimg(vul, vul, vul, 0);
  // CHECK: call <2 x i64> @llvm.s390.verimg(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}, i32 0)
  vul = __builtin_s390_verimg(vul, vul, vul, 255);
  // CHECK: call <2 x i64> @llvm.s390.verimg(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}, i32 255)

  vuc = __builtin_s390_verllb(vuc, len);
  // CHECK: call <16 x i8> @llvm.s390.verllb(<16 x i8> %{{.*}}, i32 %{{.*}})
  vus = __builtin_s390_verllh(vus, len);
  // CHECK: call <8 x i16> @llvm.s390.verllh(<8 x i16> %{{.*}}, i32 %{{.*}})
  vui = __builtin_s390_verllf(vui, len);
  // CHECK: call <4 x i32> @llvm.s390.verllf(<4 x i32> %{{.*}}, i32 %{{.*}})
  vul = __builtin_s390_verllg(vul, len);
  // CHECK: call <2 x i64> @llvm.s390.verllg(<2 x i64> %{{.*}}, i32 %{{.*}})

  vuc = __builtin_s390_verllvb(vuc, vuc);
  // CHECK: call <16 x i8> @llvm.s390.verllvb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  vus = __builtin_s390_verllvh(vus, vus);
  // CHECK: call <8 x i16> @llvm.s390.verllvh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  vui = __builtin_s390_verllvf(vui, vui);
  // CHECK: call <4 x i32> @llvm.s390.verllvf(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  vul = __builtin_s390_verllvg(vul, vul);
  // CHECK: call <2 x i64> @llvm.s390.verllvg(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})

  vus = __builtin_s390_vgfmb(vuc, vuc);
  // CHECK: call <8 x i16> @llvm.s390.vgfmb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  vui = __builtin_s390_vgfmh(vus, vus);
  // CHECK: call <4 x i32> @llvm.s390.vgfmh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  vul = __builtin_s390_vgfmf(vui, vui);
  // CHECK: call <2 x i64> @llvm.s390.vgfmf(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  vuc = __builtin_s390_vgfmg(vul, vul);
  // CHECK: call <16 x i8> @llvm.s390.vgfmg(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})

  vus = __builtin_s390_vgfmab(vuc, vuc, vus);
  // CHECK: call <8 x i16> @llvm.s390.vgfmab(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <8 x i16> %{{.*}})
  vui = __builtin_s390_vgfmah(vus, vus, vui);
  // CHECK: call <4 x i32> @llvm.s390.vgfmah(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <4 x i32> %{{.*}})
  vul = __builtin_s390_vgfmaf(vui, vui, vul);
  // CHECK: call <2 x i64> @llvm.s390.vgfmaf(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <2 x i64> %{{.*}})
  vuc = __builtin_s390_vgfmag(vul, vul, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vgfmag(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <16 x i8> %{{.*}})

  vsc = __builtin_s390_vmahb(vsc, vsc, vsc);
  // CHECK: call <16 x i8> @llvm.s390.vmahb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  vss = __builtin_s390_vmahh(vss, vss, vss);
  // CHECK: call <8 x i16> @llvm.s390.vmahh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  vsi = __builtin_s390_vmahf(vsi, vsi, vsi);
  // CHECK: call <4 x i32> @llvm.s390.vmahf(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  vuc = __builtin_s390_vmalhb(vuc, vuc, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vmalhb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  vus = __builtin_s390_vmalhh(vus, vus, vus);
  // CHECK: call <8 x i16> @llvm.s390.vmalhh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  vui = __builtin_s390_vmalhf(vui, vui, vui);
  // CHECK: call <4 x i32> @llvm.s390.vmalhf(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}})

  vss = __builtin_s390_vmaeb(vsc, vsc, vss);
  // CHECK: call <8 x i16> @llvm.s390.vmaeb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <8 x i16> %{{.*}})
  vsi = __builtin_s390_vmaeh(vss, vss, vsi);
  // CHECK: call <4 x i32> @llvm.s390.vmaeh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <4 x i32> %{{.*}})
  vsl = __builtin_s390_vmaef(vsi, vsi, vsl);
  // CHECK: call <2 x i64> @llvm.s390.vmaef(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <2 x i64> %{{.*}})
  vus = __builtin_s390_vmaleb(vuc, vuc, vus);
  // CHECK: call <8 x i16> @llvm.s390.vmaleb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <8 x i16> %{{.*}})
  vui = __builtin_s390_vmaleh(vus, vus, vui);
  // CHECK: call <4 x i32> @llvm.s390.vmaleh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <4 x i32> %{{.*}})
  vul = __builtin_s390_vmalef(vui, vui, vul);
  // CHECK: call <2 x i64> @llvm.s390.vmalef(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <2 x i64> %{{.*}})

  vss = __builtin_s390_vmaob(vsc, vsc, vss);
  // CHECK: call <8 x i16> @llvm.s390.vmaob(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <8 x i16> %{{.*}})
  vsi = __builtin_s390_vmaoh(vss, vss, vsi);
  // CHECK: call <4 x i32> @llvm.s390.vmaoh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <4 x i32> %{{.*}})
  vsl = __builtin_s390_vmaof(vsi, vsi, vsl);
  // CHECK: call <2 x i64> @llvm.s390.vmaof(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <2 x i64> %{{.*}})
  vus = __builtin_s390_vmalob(vuc, vuc, vus);
  // CHECK: call <8 x i16> @llvm.s390.vmalob(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <8 x i16> %{{.*}})
  vui = __builtin_s390_vmaloh(vus, vus, vui);
  // CHECK: call <4 x i32> @llvm.s390.vmaloh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <4 x i32> %{{.*}})
  vul = __builtin_s390_vmalof(vui, vui, vul);
  // CHECK: call <2 x i64> @llvm.s390.vmalof(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <2 x i64> %{{.*}})

  vsc = __builtin_s390_vmhb(vsc, vsc);
  // CHECK: call <16 x i8> @llvm.s390.vmhb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  vss = __builtin_s390_vmhh(vss, vss);
  // CHECK: call <8 x i16> @llvm.s390.vmhh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  vsi = __builtin_s390_vmhf(vsi, vsi);
  // CHECK: call <4 x i32> @llvm.s390.vmhf(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  vuc = __builtin_s390_vmlhb(vuc, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vmlhb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  vus = __builtin_s390_vmlhh(vus, vus);
  // CHECK: call <8 x i16> @llvm.s390.vmlhh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  vui = __builtin_s390_vmlhf(vui, vui);
  // CHECK: call <4 x i32> @llvm.s390.vmlhf(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})

  vss = __builtin_s390_vmeb(vsc, vsc);
  // CHECK: call <8 x i16> @llvm.s390.vmeb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  vsi = __builtin_s390_vmeh(vss, vss);
  // CHECK: call <4 x i32> @llvm.s390.vmeh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  vsl = __builtin_s390_vmef(vsi, vsi);
  // CHECK: call <2 x i64> @llvm.s390.vmef(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  vus = __builtin_s390_vmleb(vuc, vuc);
  // CHECK: call <8 x i16> @llvm.s390.vmleb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  vui = __builtin_s390_vmleh(vus, vus);
  // CHECK: call <4 x i32> @llvm.s390.vmleh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  vul = __builtin_s390_vmlef(vui, vui);
  // CHECK: call <2 x i64> @llvm.s390.vmlef(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})

  vss = __builtin_s390_vmob(vsc, vsc);
  // CHECK: call <8 x i16> @llvm.s390.vmob(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  vsi = __builtin_s390_vmoh(vss, vss);
  // CHECK: call <4 x i32> @llvm.s390.vmoh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  vsl = __builtin_s390_vmof(vsi, vsi);
  // CHECK: call <2 x i64> @llvm.s390.vmof(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  vus = __builtin_s390_vmlob(vuc, vuc);
  // CHECK: call <8 x i16> @llvm.s390.vmlob(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  vui = __builtin_s390_vmloh(vus, vus);
  // CHECK: call <4 x i32> @llvm.s390.vmloh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  vul = __builtin_s390_vmlof(vui, vui);
  // CHECK: call <2 x i64> @llvm.s390.vmlof(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})

  vuc = __builtin_s390_vpopctb(vuc);
  // CHECK: call <16 x i8> @llvm.ctpop.v16i8(<16 x i8> %{{.*}})
  vus = __builtin_s390_vpopcth(vus);
  // CHECK: call <8 x i16> @llvm.ctpop.v8i16(<8 x i16> %{{.*}})
  vui = __builtin_s390_vpopctf(vui);
  // CHECK: call <4 x i32> @llvm.ctpop.v4i32(<4 x i32> %{{.*}})
  vul = __builtin_s390_vpopctg(vul);
  // CHECK: call <2 x i64> @llvm.ctpop.v2i64(<2 x i64> %{{.*}})

  vuc = __builtin_s390_vsq(vuc, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vsq(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  vuc = __builtin_s390_vsbiq(vuc, vuc, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vsbiq(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  vuc = __builtin_s390_vscbiq(vuc, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vscbiq(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  vuc = __builtin_s390_vsbcbiq(vuc, vuc, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vsbcbiq(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}})

  vuc = __builtin_s390_vscbib(vuc, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vscbib(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  vus = __builtin_s390_vscbih(vus, vus);
  // CHECK: call <8 x i16> @llvm.s390.vscbih(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  vui = __builtin_s390_vscbif(vui, vui);
  // CHECK: call <4 x i32> @llvm.s390.vscbif(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  vul = __builtin_s390_vscbig(vul, vul);
  // CHECK: call <2 x i64> @llvm.s390.vscbig(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})

  vuc = __builtin_s390_vsldb(vuc, vuc, 0);
  // CHECK: call <16 x i8> @llvm.s390.vsldb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  vuc = __builtin_s390_vsldb(vuc, vuc, 15);
  // CHECK: call <16 x i8> @llvm.s390.vsldb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 15)

  vuc = __builtin_s390_vsl(vuc, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vsl(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  vuc = __builtin_s390_vslb(vuc, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vslb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})

  vuc = __builtin_s390_vsra(vuc, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vsra(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  vuc = __builtin_s390_vsrab(vuc, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vsrab(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})

  vuc = __builtin_s390_vsrl(vuc, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vsrl(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  vuc = __builtin_s390_vsrlb(vuc, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vsrlb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})

  vui = __builtin_s390_vsumb(vuc, vuc);
  // CHECK: call <4 x i32> @llvm.s390.vsumb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  vui = __builtin_s390_vsumh(vus, vus);
  // CHECK: call <4 x i32> @llvm.s390.vsumh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  vul = __builtin_s390_vsumgh(vus, vus);
  // CHECK: call <2 x i64> @llvm.s390.vsumgh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  vul = __builtin_s390_vsumgf(vui, vui);
  // CHECK: call <2 x i64> @llvm.s390.vsumgf(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  vuc = __builtin_s390_vsumqf(vui, vui);
  // CHECK: call <16 x i8> @llvm.s390.vsumqf(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  vuc = __builtin_s390_vsumqg(vul, vul);
  // CHECK: call <16 x i8> @llvm.s390.vsumqg(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})

  len = __builtin_s390_vtm(vuc, vuc);
  // CHECK: call i32 @llvm.s390.vtm(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})

  vsc = __builtin_s390_vceqbs(vsc, vsc, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vceqbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  vss = __builtin_s390_vceqhs(vss, vss, &cc);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vceqhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  vsi = __builtin_s390_vceqfs(vsi, vsi, &cc);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vceqfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  vsl = __builtin_s390_vceqgs(vsl, vsl, &cc);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vceqgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})

  vsc = __builtin_s390_vchbs(vsc, vsc, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vchbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  vss = __builtin_s390_vchhs(vss, vss, &cc);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vchhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  vsi = __builtin_s390_vchfs(vsi, vsi, &cc);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vchfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  vsl = __builtin_s390_vchgs(vsl, vsl, &cc);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vchgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})

  vsc = __builtin_s390_vchlbs(vuc, vuc, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vchlbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  vss = __builtin_s390_vchlhs(vus, vus, &cc);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vchlhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  vsi = __builtin_s390_vchlfs(vui, vui, &cc);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vchlfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  vsl = __builtin_s390_vchlgs(vul, vul, &cc);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vchlgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
}

void test_string(void) {
  vuc = __builtin_s390_vfaeb(vuc, vuc, 0);
  // CHECK: call <16 x i8> @llvm.s390.vfaeb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  vuc = __builtin_s390_vfaeb(vuc, vuc, 15);
  // CHECK: call <16 x i8> @llvm.s390.vfaeb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 15)
  vus = __builtin_s390_vfaeh(vus, vus, 0);
  // CHECK: call <8 x i16> @llvm.s390.vfaeh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i32 0)
  vus = __builtin_s390_vfaeh(vus, vus, 15);
  // CHECK: call <8 x i16> @llvm.s390.vfaeh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i32 15)
  vui = __builtin_s390_vfaef(vui, vui, 0);
  // CHECK: call <4 x i32> @llvm.s390.vfaef(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 0)
  vui = __builtin_s390_vfaef(vui, vui, 15);
  // CHECK: call <4 x i32> @llvm.s390.vfaef(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 15)

  vuc = __builtin_s390_vfaezb(vuc, vuc, 0);
  // CHECK: call <16 x i8> @llvm.s390.vfaezb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  vuc = __builtin_s390_vfaezb(vuc, vuc, 15);
  // CHECK: call <16 x i8> @llvm.s390.vfaezb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 15)
  vus = __builtin_s390_vfaezh(vus, vus, 0);
  // CHECK: call <8 x i16> @llvm.s390.vfaezh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i32 0)
  vus = __builtin_s390_vfaezh(vus, vus, 15);
  // CHECK: call <8 x i16> @llvm.s390.vfaezh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i32 15)
  vui = __builtin_s390_vfaezf(vui, vui, 0);
  // CHECK: call <4 x i32> @llvm.s390.vfaezf(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 0)
  vui = __builtin_s390_vfaezf(vui, vui, 15);
  // CHECK: call <4 x i32> @llvm.s390.vfaezf(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 15)

  vuc = __builtin_s390_vfeeb(vuc, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vfeeb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  vus = __builtin_s390_vfeeh(vus, vus);
  // CHECK: call <8 x i16> @llvm.s390.vfeeh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  vui = __builtin_s390_vfeef(vui, vui);
  // CHECK: call <4 x i32> @llvm.s390.vfeef(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})

  vuc = __builtin_s390_vfeezb(vuc, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vfeezb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  vus = __builtin_s390_vfeezh(vus, vus);
  // CHECK: call <8 x i16> @llvm.s390.vfeezh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  vui = __builtin_s390_vfeezf(vui, vui);
  // CHECK: call <4 x i32> @llvm.s390.vfeezf(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})

  vuc = __builtin_s390_vfeneb(vuc, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vfeneb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  vus = __builtin_s390_vfeneh(vus, vus);
  // CHECK: call <8 x i16> @llvm.s390.vfeneh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  vui = __builtin_s390_vfenef(vui, vui);
  // CHECK: call <4 x i32> @llvm.s390.vfenef(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})

  vuc = __builtin_s390_vfenezb(vuc, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vfenezb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  vus = __builtin_s390_vfenezh(vus, vus);
  // CHECK: call <8 x i16> @llvm.s390.vfenezh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  vui = __builtin_s390_vfenezf(vui, vui);
  // CHECK: call <4 x i32> @llvm.s390.vfenezf(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})

  vuc = __builtin_s390_vistrb(vuc);
  // CHECK: call <16 x i8> @llvm.s390.vistrb(<16 x i8> %{{.*}})
  vus = __builtin_s390_vistrh(vus);
  // CHECK: call <8 x i16> @llvm.s390.vistrh(<8 x i16> %{{.*}})
  vui = __builtin_s390_vistrf(vui);
  // CHECK: call <4 x i32> @llvm.s390.vistrf(<4 x i32> %{{.*}})

  vuc = __builtin_s390_vstrcb(vuc, vuc, vuc, 0);
  // CHECK: call <16 x i8> @llvm.s390.vstrcb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  vuc = __builtin_s390_vstrcb(vuc, vuc, vuc, 15);
  // CHECK: call <16 x i8> @llvm.s390.vstrcb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 15)
  vus = __builtin_s390_vstrch(vus, vus, vus, 0);
  // CHECK: call <8 x i16> @llvm.s390.vstrch(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i32 0)
  vus = __builtin_s390_vstrch(vus, vus, vus, 15);
  // CHECK: call <8 x i16> @llvm.s390.vstrch(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i32 15)
  vui = __builtin_s390_vstrcf(vui, vui, vui, 0);
  // CHECK: call <4 x i32> @llvm.s390.vstrcf(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 0)
  vui = __builtin_s390_vstrcf(vui, vui, vui, 15);
  // CHECK: call <4 x i32> @llvm.s390.vstrcf(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 15)

  vuc = __builtin_s390_vstrczb(vuc, vuc, vuc, 0);
  // CHECK: call <16 x i8> @llvm.s390.vstrczb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  vuc = __builtin_s390_vstrczb(vuc, vuc, vuc, 15);
  // CHECK: call <16 x i8> @llvm.s390.vstrczb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 15)
  vus = __builtin_s390_vstrczh(vus, vus, vus, 0);
  // CHECK: call <8 x i16> @llvm.s390.vstrczh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i32 0)
  vus = __builtin_s390_vstrczh(vus, vus, vus, 15);
  // CHECK: call <8 x i16> @llvm.s390.vstrczh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i32 15)
  vui = __builtin_s390_vstrczf(vui, vui, vui, 0);
  // CHECK: call <4 x i32> @llvm.s390.vstrczf(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 0)
  vui = __builtin_s390_vstrczf(vui, vui, vui, 15);
  // CHECK: call <4 x i32> @llvm.s390.vstrczf(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 15)

  vuc = __builtin_s390_vfaebs(vuc, vuc, 0, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vfaebs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  vuc = __builtin_s390_vfaebs(vuc, vuc, 15, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vfaebs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 15)
  vus = __builtin_s390_vfaehs(vus, vus, 0, &cc);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vfaehs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i32 0)
  vus = __builtin_s390_vfaehs(vus, vus, 15, &cc);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vfaehs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i32 15)
  vui = __builtin_s390_vfaefs(vui, vui, 0, &cc);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vfaefs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 0)
  vui = __builtin_s390_vfaefs(vui, vui, 15, &cc);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vfaefs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 15)

  vuc = __builtin_s390_vfaezbs(vuc, vuc, 0, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vfaezbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  vuc = __builtin_s390_vfaezbs(vuc, vuc, 15, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vfaezbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 15)
  vus = __builtin_s390_vfaezhs(vus, vus, 0, &cc);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vfaezhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i32 0)
  vus = __builtin_s390_vfaezhs(vus, vus, 15, &cc);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vfaezhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i32 15)
  vui = __builtin_s390_vfaezfs(vui, vui, 0, &cc);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vfaezfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 0)
  vui = __builtin_s390_vfaezfs(vui, vui, 15, &cc);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vfaezfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 15)

  vuc = __builtin_s390_vfeebs(vuc, vuc, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vfeebs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  vus = __builtin_s390_vfeehs(vus, vus, &cc);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vfeehs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  vui = __builtin_s390_vfeefs(vui, vui, &cc);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vfeefs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})

  vuc = __builtin_s390_vfeezbs(vuc, vuc, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vfeezbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  vus = __builtin_s390_vfeezhs(vus, vus, &cc);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vfeezhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  vui = __builtin_s390_vfeezfs(vui, vui, &cc);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vfeezfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})

  vuc = __builtin_s390_vfenebs(vuc, vuc, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vfenebs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  vus = __builtin_s390_vfenehs(vus, vus, &cc);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vfenehs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  vui = __builtin_s390_vfenefs(vui, vui, &cc);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vfenefs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})

  vuc = __builtin_s390_vfenezbs(vuc, vuc, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vfenezbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  vus = __builtin_s390_vfenezhs(vus, vus, &cc);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vfenezhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  vui = __builtin_s390_vfenezfs(vui, vui, &cc);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vfenezfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})

  vuc = __builtin_s390_vistrbs(vuc, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vistrbs(<16 x i8> %{{.*}})
  vus = __builtin_s390_vistrhs(vus, &cc);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vistrhs(<8 x i16> %{{.*}})
  vui = __builtin_s390_vistrfs(vui, &cc);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vistrfs(<4 x i32> %{{.*}})

  vuc = __builtin_s390_vstrcbs(vuc, vuc, vuc, 0, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vstrcbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  vuc = __builtin_s390_vstrcbs(vuc, vuc, vuc, 15, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vstrcbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 15)
  vus = __builtin_s390_vstrchs(vus, vus, vus, 0, &cc);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vstrchs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i32 0)
  vus = __builtin_s390_vstrchs(vus, vus, vus, 15, &cc);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vstrchs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i32 15)
  vui = __builtin_s390_vstrcfs(vui, vui, vui, 0, &cc);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vstrcfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 0)
  vui = __builtin_s390_vstrcfs(vui, vui, vui, 15, &cc);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vstrcfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 15)

  vuc = __builtin_s390_vstrczbs(vuc, vuc, vuc, 0, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vstrczbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  vuc = __builtin_s390_vstrczbs(vuc, vuc, vuc, 15, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vstrczbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 15)
  vus = __builtin_s390_vstrczhs(vus, vus, vus, 0, &cc);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vstrczhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i32 0)
  vus = __builtin_s390_vstrczhs(vus, vus, vus, 15, &cc);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vstrczhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i32 15)
  vui = __builtin_s390_vstrczfs(vui, vui, vui, 0, &cc);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vstrczfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 0)
  vui = __builtin_s390_vstrczfs(vui, vui, vui, 15, &cc);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vstrczfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 15)
}

void test_float(void) {
  vsl = __builtin_s390_vfcedbs(vd, vd, &cc);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vfcedbs(<2 x double> %{{.*}}, <2 x double> %{{.*}})
  vsl = __builtin_s390_vfchdbs(vd, vd, &cc);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vfchdbs(<2 x double> %{{.*}}, <2 x double> %{{.*}})
  vsl = __builtin_s390_vfchedbs(vd, vd, &cc);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vfchedbs(<2 x double> %{{.*}}, <2 x double> %{{.*}})

  vsl = __builtin_s390_vftcidb(vd, 0, &cc);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vftcidb(<2 x double> %{{.*}}, i32 0)
  vsl = __builtin_s390_vftcidb(vd, 4095, &cc);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vftcidb(<2 x double> %{{.*}}, i32 4095)

  vd = __builtin_s390_vfsqdb(vd);
  // CHECK: call <2 x double> @llvm.sqrt.v2f64(<2 x double> %{{.*}})

  vd = __builtin_s390_vfmadb(vd, vd, vd);
  // CHECK: call <2 x double> @llvm.fma.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}})
  vd = __builtin_s390_vfmsdb(vd, vd, vd);
  // CHECK: [[NEG:%[^ ]+]] = fneg <2 x double> %{{.*}}
  // CHECK: call <2 x double> @llvm.fma.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> [[NEG]])

  vd = __builtin_s390_vflpdb(vd);
  // CHECK: call <2 x double> @llvm.fabs.v2f64(<2 x double> %{{.*}})
  vd = __builtin_s390_vflndb(vd);
  // CHECK: [[ABS:%[^ ]+]] = call <2 x double> @llvm.fabs.v2f64(<2 x double> %{{.*}})
  // CHECK: fneg <2 x double> [[ABS]]

  vd = __builtin_s390_vfidb(vd, 0, 0);
  // CHECK: call <2 x double> @llvm.rint.v2f64(<2 x double> %{{.*}})
  vd = __builtin_s390_vfidb(vd, 4, 0);
  // CHECK: call <2 x double> @llvm.nearbyint.v2f64(<2 x double> %{{.*}})
  vd = __builtin_s390_vfidb(vd, 4, 1);
  // CHECK: call <2 x double> @llvm.round.v2f64(<2 x double> %{{.*}})
  vd = __builtin_s390_vfidb(vd, 4, 5);
  // CHECK: call <2 x double> @llvm.trunc.v2f64(<2 x double> %{{.*}})
  vd = __builtin_s390_vfidb(vd, 4, 6);
  // CHECK: call <2 x double> @llvm.ceil.v2f64(<2 x double> %{{.*}})
  vd = __builtin_s390_vfidb(vd, 4, 7);
  // CHECK: call <2 x double> @llvm.floor.v2f64(<2 x double> %{{.*}})
  vd = __builtin_s390_vfidb(vd, 4, 4);
  // CHECK: call <2 x double> @llvm.s390.vfidb(<2 x double> %{{.*}}, i32 4, i32 4)
}

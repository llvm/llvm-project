// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -target-feature +altivec -target-feature +power8-vector \
// RUN: -triple powerpc64-unknown-unknown -emit-llvm %s -o - | FileCheck %s

// RUN: %clang_cc1 -target-feature +altivec -target-feature +power8-vector \
// RUN: -triple powerpc64le-unknown-unknown -emit-llvm %s -o - \
// RUN: | FileCheck %s -check-prefix=CHECK-LE

// RUN: not %clang_cc1 -target-feature +altivec -triple powerpc-unknown-unknown \
// RUN: -emit-llvm %s -o - 2>&1 | FileCheck %s -check-prefix=CHECK-PPC
#include <altivec.h>

// CHECK-PPC: error: __int128 is not supported on this target
vector signed __int128 vlll = { -1 };
// CHECK-PPC: error: __int128 is not supported on this target
vector unsigned __int128 vulll = { 1 };

signed long long param_sll;
// CHECK-PPC: error: __int128 is not supported on this target
signed __int128 param_lll;
// CHECK-PPC: error: __int128 is not supported on this target
unsigned __int128 param_ulll;

// CHECK-PPC: error: __int128 is not supported on this target
vector signed __int128 res_vlll;
// CHECK-PPC: error: __int128 is not supported on this target
vector unsigned __int128 res_vulll;


// CHECK-LABEL: define{{.*}} void @test1
void test1() {

  /* vec_add */
  res_vlll = vec_add(vlll, vlll);
// CHECK: add <1 x i128>
// CHECK-LE: add <1 x i128> 
// CHECK-PPC: error: call to 'vec_add' is ambiguous

  res_vulll = vec_add(vulll, vulll);
// CHECK: add <1 x i128> 
// CHECK-LE: add <1 x i128> 
// CHECK-PPC: error: call to 'vec_add' is ambiguous

  /* vec_vadduqm */
  res_vlll = vec_vadduqm(vlll, vlll);
// CHECK: add <1 x i128> 
// CHECK-LE: add <1 x i128> 
// CHECK-PPC: error: assigning to '__vector __int128' (vector of 1 '__int128' value) from incompatible type 'int'

  res_vulll = vec_vadduqm(vulll, vulll);
// CHECK: add <1 x i128> 
// CHECK-LE: add <1 x i128> 
// CHECK-PPC: error: assigning to '__vector unsigned __int128' (vector of 1 'unsigned __int128' value) from incompatible type 'int'

  /* vec_vaddeuqm */
  res_vlll = vec_vaddeuqm(vlll, vlll, vlll);
// CHECK: @llvm.ppc.altivec.vaddeuqm
// CHECK-LE: @llvm.ppc.altivec.vaddeuqm
// CHECK-PPC: error: assigning to '__vector __int128' (vector of 1 '__int128' value) from incompatible type 'int'
  
  res_vulll = vec_vaddeuqm(vulll, vulll, vulll);
// CHECK: @llvm.ppc.altivec.vaddeuqm
// CHECK-LE: @llvm.ppc.altivec.vaddeuqm
// CHECK-PPC: error: assigning to '__vector unsigned __int128' (vector of 1 'unsigned __int128' value) from incompatible type 'int'

  /* vec_addc */
  res_vlll = vec_addc(vlll, vlll);
// CHECK: @llvm.ppc.altivec.vaddcuq
// CHECK-LE: @llvm.ppc.altivec.vaddcuq
// KCHECK-PPC: error: call to 'vec_addc' is ambiguous

  res_vulll = vec_addc(vulll, vulll);
// CHECK: @llvm.ppc.altivec.vaddcuq
// CHECK-LE: @llvm.ppc.altivec.vaddcuq
// KCHECK-PPC: error: call to 'vec_addc' is ambiguous


  /* vec_vaddcuq */
  res_vlll = vec_vaddcuq(vlll, vlll);
// CHECK: @llvm.ppc.altivec.vaddcuq
// CHECK-LE: @llvm.ppc.altivec.vaddcuq
// CHECK-PPC: error: assigning to '__vector __int128' (vector of 1 '__int128' value) from incompatible type 'int'
  
  res_vulll = vec_vaddcuq(vulll, vulll);
// CHECK: @llvm.ppc.altivec.vaddcuq
// CHECK-LE: @llvm.ppc.altivec.vaddcuq
// CHECK-PPC: error: assigning to '__vector unsigned __int128' (vector of 1 'unsigned __int128' value) from incompatible type 'int'

  /* vec_vaddecuq */
  res_vlll = vec_vaddecuq(vlll, vlll, vlll);
// CHECK: @llvm.ppc.altivec.vaddecuq
// CHECK-LE: @llvm.ppc.altivec.vaddecuq
// CHECK-PPC: error: assigning to '__vector __int128' (vector of 1 '__int128' value) from incompatible type 'int'

  res_vulll = vec_vaddecuq(vulll, vulll, vulll);
// CHECK: @llvm.ppc.altivec.vaddecuq
// CHECK-LE: @llvm.ppc.altivec.vaddecuq
// CHECK-PPC: error: assigning to '__vector unsigned __int128' (vector of 1 'unsigned __int128' value) from incompatible type 'int'

  /* vec_sub */
  res_vlll = vec_sub(vlll, vlll);
// CHECK: sub <1 x i128>
// CHECK-LE: sub <1 x i128> 
// CHECK-PPC: error: call to 'vec_sub' is ambiguous
  
  res_vulll = vec_sub(vulll, vulll);
// CHECK: sub <1 x i128> 
// CHECK-LE: sub <1 x i128> 
// CHECK-PPC: error: call to 'vec_sub' is ambiguous

  /* vec_vsubuqm */
  res_vlll = vec_vsubuqm(vlll, vlll);
// CHECK: sub <1 x i128> 
// CHECK-LE: sub <1 x i128> 
// CHECK-PPC: error: assigning to '__vector __int128' (vector of 1 '__int128' value) from incompatible type 'int'
  
  res_vulll = vec_vsubuqm(vulll, vulll);
// CHECK: sub <1 x i128> 
// CHECK-LE: sub <1 x i128> 
// CHECK-PPC: error: assigning to '__vector unsigned __int128' (vector of 1 'unsigned __int128' value) from incompatible type 'int'
  
  /* vec_vsubeuqm */
  res_vlll = vec_vsubeuqm(vlll, vlll, vlll);
// CHECK: @llvm.ppc.altivec.vsubeuqm
// CHECK-LE: @llvm.ppc.altivec.vsubeuqm
// CHECK-PPC: error: assigning to '__vector __int128' (vector of 1 '__int128' value) from incompatible type 'int'
  
  /* vec_sube */
  res_vlll = vec_sube(vlll, vlll, vlll);
// CHECK: @llvm.ppc.altivec.vsubeuqm
// CHECK-LE: @llvm.ppc.altivec.vsubeuqm
// CHECK-PPC: error: call to 'vec_sube' is ambiguous
  
  res_vulll = vec_sube(vulll, vulll, vulll);
// CHECK: @llvm.ppc.altivec.vsubeuqm
// CHECK-LE: @llvm.ppc.altivec.vsubeuqm
// CHECK-PPC: error: call to 'vec_sube' is ambiguous
  
  res_vlll = vec_sube(vlll, vlll, vlll);
// CHECK: @llvm.ppc.altivec.vsubeuqm
// CHECK-LE: @llvm.ppc.altivec.vsubeuqm
// CHECK-PPC: error: call to 'vec_sube' is ambiguous
  
  res_vulll = vec_vsubeuqm(vulll, vulll, vulll);
// CHECK: @llvm.ppc.altivec.vsubeuqm
// CHECK-LE: @llvm.ppc.altivec.vsubeuqm
// CHECK-PPC: error: assigning to '__vector unsigned __int128' (vector of 1 'unsigned __int128' value) from incompatible type 'int'
  
  res_vulll = vec_sube(vulll, vulll, vulll);
// CHECK: @llvm.ppc.altivec.vsubeuqm
// CHECK-LE: @llvm.ppc.altivec.vsubeuqm
// CHECK-PPC: error: call to 'vec_sube' is ambiguous

  /* vec_subc */
  res_vlll = vec_subc(vlll, vlll);
// CHECK: @llvm.ppc.altivec.vsubcuq
// CHECK-LE: @llvm.ppc.altivec.vsubcuq
// KCHECK-PPC: error: call to 'vec_subc' is ambiguous

  res_vulll = vec_subc(vulll, vulll);
// CHECK: @llvm.ppc.altivec.vsubcuq
// CHECK-LE: @llvm.ppc.altivec.vsubcuq
// KCHECK-PPC: error: call to 'vec_subc' is ambiguous

  /* vec_vsubcuq */
  res_vlll = vec_vsubcuq(vlll, vlll);
// CHECK: @llvm.ppc.altivec.vsubcuq
// CHECK-LE: @llvm.ppc.altivec.vsubcuq
// CHECK-PPC: error: assigning to '__vector __int128' (vector of 1 '__int128' value) from incompatible type 'int'
  
  res_vulll = vec_vsubcuq(vulll, vulll);
// CHECK: @llvm.ppc.altivec.vsubcuq
// CHECK-LE: @llvm.ppc.altivec.vsubcuq
// CHECK-PPC: error: assigning to '__vector unsigned __int128' (vector of 1 'unsigned __int128' value) from incompatible type 'int'

  /* vec_vsubecuq */
  res_vlll = vec_vsubecuq(vlll, vlll, vlll);
// CHECK: @llvm.ppc.altivec.vsubecuq
// CHECK-LE: @llvm.ppc.altivec.vsubecuq
// CHECK-PPC: error: assigning to '__vector __int128' (vector of 1 '__int128' value) from incompatible type 'int'

  res_vulll = vec_vsubecuq(vulll, vulll, vulll);
// CHECK: @llvm.ppc.altivec.vsubecuq
// CHECK-LE: @llvm.ppc.altivec.vsubecuq
// CHECK-PPC: error: assigning to '__vector unsigned __int128' (vector of 1 'unsigned __int128' value) from incompatible type 'int'

  res_vlll = vec_subec(vlll, vlll, vlll);
// CHECK: @llvm.ppc.altivec.vsubecuq
// CHECK-LE: @llvm.ppc.altivec.vsubecuq
// CHECK-PPC: error: assigning to '__vector __int128' (vector of 1 '__int128' value) from incompatible type 'int'  

  res_vulll = vec_subec(vulll, vulll, vulll);
// CHECK: @llvm.ppc.altivec.vsubecuq
// CHECK-LE: @llvm.ppc.altivec.vsubecuq
// CHECK-PPC: error: assigning to '__vector unsigned __int128' (vector of 1 'unsigned __int128' value) from incompatible type 'int'  

  res_vulll = vec_revb(vulll);
// CHECK: store <16 x i8> <i8 15, i8 14, i8 13, i8 12, i8 11, i8 10, i8 9, i8 8, i8 7, i8 6, i8 5, i8 4, i8 3, i8 2, i8 1, i8 0>, <16 x i8>* {{%.+}}, align 16
// CHECK: call <4 x i32> @llvm.ppc.altivec.vperm(<4 x i32> {{%.+}}, <4 x i32> {{%.+}}, <16 x i8> {{%.+}})
// CHECK-LE: store <16 x i8> <i8 15, i8 14, i8 13, i8 12, i8 11, i8 10, i8 9, i8 8, i8 7, i8 6, i8 5, i8 4, i8 3, i8 2, i8 1, i8 0>, <16 x i8>* {{%.+}}, align 16
// CHECK-LE: store <16 x i8> <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>, <16 x i8>* {{%.+}}, align 16
// CHECK-LE: xor <16 x i8>
// CHECK-LE: call <4 x i32> @llvm.ppc.altivec.vperm(<4 x i32> {{%.+}}, <4 x i32> {{%.+}}, <16 x i8> {{%.+}})
// CHECK_PPC: error: call to 'vec_revb' is ambiguous

  /* vec_xl */
  res_vlll = vec_xl(param_sll, &param_lll);
  // CHECK: load <1 x i128>, <1 x i128>* %{{[0-9]+}}, align 1
  // CHECK-LE: load <1 x i128>, <1 x i128>* %{{[0-9]+}}, align 1
  // CHECK-PPC: error: call to 'vec_xl' is ambiguous

  res_vulll = vec_xl(param_sll, &param_ulll);
  // CHECK: load <1 x i128>, <1 x i128>* %{{[0-9]+}}, align 1
  // CHECK-LE: load <1 x i128>, <1 x i128>* %{{[0-9]+}}, align 1
  // CHECK-PPC: error: call to 'vec_xl' is ambiguous

  /* vec_xst */
   vec_xst(vlll, param_sll, &param_lll);
  // CHECK: store <1 x i128> %{{[0-9]+}}, <1 x i128>* %{{[0-9]+}}, align 1
  // CHECK-LE: store <1 x i128> %{{[0-9]+}}, <1 x i128>* %{{[0-9]+}}, align 1
  // CHECK-PPC: error: call to 'vec_xst' is ambiguous

   vec_xst(vulll, param_sll, &param_ulll);
  // CHECK: store <1 x i128> %{{[0-9]+}}, <1 x i128>* %{{[0-9]+}}, align 1
  // CHECK-LE: store <1 x i128> %{{[0-9]+}}, <1 x i128>* %{{[0-9]+}}, align 1
  // CHECK-PPC: error: call to 'vec_xst' is ambiguous

  /* vec_xl_be */
  res_vlll = vec_xl_be(param_sll, &param_lll);
  // CHECK: load <1 x i128>, <1 x i128>* %{{[0-9]+}}, align 1
  // CHECK-LE: load <1 x i128>, <1 x i128>* %{{[0-9]+}}, align 1
  // CHECK-PPC: error: call to 'vec_xl' is ambiguous

  res_vulll = vec_xl_be(param_sll, &param_ulll);
  // CHECK: load <1 x i128>, <1 x i128>* %{{[0-9]+}}, align 1
  // CHECK-LE: load <1 x i128>, <1 x i128>* %{{[0-9]+}}, align 1
  // CHECK-PPC: error: call to 'vec_xl' is ambiguous

  /* vec_xst_be  */
   vec_xst_be(vlll, param_sll, &param_lll);
  // CHECK: store <1 x i128> %{{[0-9]+}}, <1 x i128>* %{{[0-9]+}}, align 1
  // CHECK-LE: store <1 x i128> %{{[0-9]+}}, <1 x i128>* %{{[0-9]+}}, align 1
  // CHECK-PPC: error: call to 'vec_xst' is ambiguous

   vec_xst_be(vulll, param_sll, &param_ulll);
  // CHECK: store <1 x i128> %{{[0-9]+}}, <1 x i128>* %{{[0-9]+}}, align 1
  // CHECK-LE: store <1 x i128> %{{[0-9]+}}, <1 x i128>* %{{[0-9]+}}, align 1
  // CHECK-PPC: error: call to 'vec_xst' is ambiguous
}

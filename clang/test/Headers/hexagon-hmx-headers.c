// RUN: %clang_cc1 -O0 -internal-isystem %S/../../lib/Headers/ \
// RUN:   -target-cpu hexagonv79 -triple hexagon-unknown-elf \
// RUN:   -target-feature +hmx -emit-llvm -o - %s | FileCheck %s

#include <hmx_hexagon_protos.h>

// CHECK-LABEL: @test_Q6_mxclracc_hf
// CHECK: call void @llvm.hexagon.M8.mxclracc.hf()
void test_Q6_mxclracc_hf(void) {
  Q6_mxclracc_hf();
}

// CHECK-LABEL: @test_Q6_bias_mxmem2_A
// CHECK: call void @llvm.hexagon.M8.mxmem2.bias(ptr %{{[0-9]+}})
void test_Q6_bias_mxmem2_A(void *rs) {
  Q6_bias_mxmem2_A(rs);
}

// CHECK-LABEL: @test_Q6_mxmem2_bias_A
// CHECK: call void @llvm.hexagon.M8.mxmem2.st.bias(ptr %{{[0-9]+}})
void test_Q6_mxmem2_bias_A(void *rs) {
  Q6_mxmem2_bias_A(rs);
}

// CHECK-LABEL: @test_Q6_activation_hf_mxmem_RR(
// CHECK: call void @llvm.hexagon.M8.mxmem.blk.sm.act.hf(i32 %{{[0-9]+}}, i32 %{{[0-9]+}})
void test_Q6_activation_hf_mxmem_RR(unsigned int rs, int rt) {
  Q6_activation_hf_mxmem_RR(rs, rt);
}

// CHECK-LABEL: @test_Q6_activation_hf_mxmem_RR_deep(
// CHECK: call void @llvm.hexagon.M8.mxmem.sm.act.hf(i32 %{{[0-9]+}}, i32 %{{[0-9]+}})
void test_Q6_activation_hf_mxmem_RR_deep(unsigned int rs, int rt) {
  Q6_activation_hf_mxmem_RR_deep(rs, rt);
}

// CHECK-LABEL: @test_Q6_weight_hf_mxmem_RR(
// CHECK: call void @llvm.hexagon.M8.mxmem.wei.hf(i32 %{{[0-9]+}}, i32 %{{[0-9]+}})
void test_Q6_weight_hf_mxmem_RR(unsigned int rs, int rt) {
  Q6_weight_hf_mxmem_RR(rs, rt);
}

// CHECK-LABEL: @test_Q6_weight_hf_mxmem_RR_deep(
// CHECK: call void @llvm.hexagon.M8.mxmemdp.wei.hf(i32 %{{[0-9]+}}, i32 %{{[0-9]+}})
void test_Q6_weight_hf_mxmem_RR_deep(unsigned int rs, int rt) {
  Q6_weight_hf_mxmem_RR_deep(rs, rt);
}

// CHECK-LABEL: @test_Q6_activation_hf_mxmem_RR_single(
// CHECK: call void @llvm.hexagon.M8.mxmems.blk.sm.act.hf(i32 %{{[0-9]+}}, i32 %{{[0-9]+}})
void test_Q6_activation_hf_mxmem_RR_single(unsigned int rs, int rt) {
  Q6_activation_hf_mxmem_RR_single(rs, rt);
}

// CHECK-LABEL: @test_Q6_cvt_hf_acc_R(
// CHECK: call void @llvm.hexagon.M8.cvt.rs.hf(i32 %{{[0-9]+}})
void test_Q6_cvt_hf_acc_R(int rs) {
  Q6_cvt_hf_acc_R(rs);
}

// CHECK-LABEL: @test_Q6_mxmem_cvt_RR(
// CHECK: call void @llvm.hexagon.M8.mxmem(ptr %{{[0-9]+}}, i32 %{{[0-9]+}})
void test_Q6_mxmem_cvt_RR(void *rs, int rt) {
  Q6_mxmem_cvt_RR(rs, rt);
}

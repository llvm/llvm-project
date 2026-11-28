// RUN: %clang_cc1 -triple hexagon -target-cpu hexagonv73 -target-feature +hmx \
// RUN:   -emit-llvm -o - %s | FileCheck %s

// Test that each HMX builtin is lowered to the corresponding LLVM intrinsic.

// CHECK-LABEL: @test_M8_mxclracc_hf
// CHECK: call void @llvm.hexagon.M8.mxclracc.hf()
void test_M8_mxclracc_hf(void) {
  __builtin_HEXAGON_M8_mxclracc_hf();
}

// CHECK-LABEL: @test_M8_mxmem2_bias
// CHECK: call void @llvm.hexagon.M8.mxmem2.bias(ptr %{{[0-9]+}})
void test_M8_mxmem2_bias(void *a) {
  __builtin_HEXAGON_M8_mxmem2_bias(a);
}

// CHECK-LABEL: @test_M8_mxmem2_st_bias
// CHECK: call void @llvm.hexagon.M8.mxmem2.st.bias(ptr %{{[0-9]+}})
void test_M8_mxmem2_st_bias(void *a) {
  __builtin_HEXAGON_M8_mxmem2_st_bias(a);
}

// CHECK-LABEL: @test_M8_mxmem_blk_sm_act_hf
// CHECK: call void @llvm.hexagon.M8.mxmem.blk.sm.act.hf(i32 %{{[0-9]+}}, i32 %{{[0-9]+}})
void test_M8_mxmem_blk_sm_act_hf(unsigned int a, int b) {
  __builtin_HEXAGON_M8_mxmem_blk_sm_act_hf(a, b);
}

// CHECK-LABEL: @test_M8_mxmem_sm_act_hf
// CHECK: call void @llvm.hexagon.M8.mxmem.sm.act.hf(i32 %{{[0-9]+}}, i32 %{{[0-9]+}})
void test_M8_mxmem_sm_act_hf(unsigned int a, int b) {
  __builtin_HEXAGON_M8_mxmem_sm_act_hf(a, b);
}

// CHECK-LABEL: @test_M8_mxmem_wei_hf
// CHECK: call void @llvm.hexagon.M8.mxmem.wei.hf(i32 %{{[0-9]+}}, i32 %{{[0-9]+}})
void test_M8_mxmem_wei_hf(unsigned int a, int b) {
  __builtin_HEXAGON_M8_mxmem_wei_hf(a, b);
}

// CHECK-LABEL: @test_M8_mxmemdp_wei_hf
// CHECK: call void @llvm.hexagon.M8.mxmemdp.wei.hf(i32 %{{[0-9]+}}, i32 %{{[0-9]+}})
void test_M8_mxmemdp_wei_hf(unsigned int a, int b) {
  __builtin_HEXAGON_M8_mxmemdp_wei_hf(a, b);
}

// CHECK-LABEL: @test_M8_mxmems_blk_sm_act_hf
// CHECK: call void @llvm.hexagon.M8.mxmems.blk.sm.act.hf(i32 %{{[0-9]+}}, i32 %{{[0-9]+}})
void test_M8_mxmems_blk_sm_act_hf(unsigned int a, int b) {
  __builtin_HEXAGON_M8_mxmems_blk_sm_act_hf(a, b);
}

// CHECK-LABEL: @test_M8_cvt_rs_hf
// CHECK: call void @llvm.hexagon.M8.cvt.rs.hf(i32 %{{[0-9]+}})
void test_M8_cvt_rs_hf(int a) {
  __builtin_HEXAGON_M8_cvt_rs_hf(a);
}

// CHECK-LABEL: @test_M8_mxmem
// CHECK: call void @llvm.hexagon.M8.mxmem(ptr %{{[0-9]+}}, i32 %{{[0-9]+}})
void test_M8_mxmem(void *a, int b) {
  __builtin_HEXAGON_M8_mxmem(a, b);
}

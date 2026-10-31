// RUN: inter-opt %s --inter-canonicalize-dpas-builtin | FileCheck %s

module {
  llvm.func @_Z38intel_sub_group_f16_f16_matrix_mad_k16Dv8_sDv8_iDv8_f(vector<8xi16>, vector<8xi32>, vector<8xf32>) -> vector<8xf32>
  llvm.func @_Z40intel_sub_group_bf16_bf16_matrix_mad_k16Dv8_sDv8_iDv8_f(vector<8xi16>, vector<8xi32>, vector<8xf32>) -> vector<8xf32>

  func.func @dpas(%a: vector<8xi16>, %b: vector<8xi32>, %acc: vector<8xf32>)
      attributes {xw.simd_width = 16 : i32} {
    %f16 = llvm.call @_Z38intel_sub_group_f16_f16_matrix_mad_k16Dv8_sDv8_iDv8_f(%a, %b, %acc) : (vector<8xi16>, vector<8xi32>, vector<8xf32>) -> vector<8xf32>
    %bf16 = llvm.call @_Z40intel_sub_group_bf16_bf16_matrix_mad_k16Dv8_sDv8_iDv8_f(%a, %b, %f16) : (vector<8xi16>, vector<8xi32>, vector<8xf32>) -> vector<8xf32>
    return
  }
}

// CHECK-LABEL: func.func @dpas
// CHECK: xw.dpas {{.*}}a_precision = 0 : i32{{.*}}b_precision = 0 : i32{{.*}}k = 16 : i64{{.*}}repeat_count = 8 : i64{{.*}}systolic_depth = 8 : i64
// CHECK: xw.dpas {{.*}}a_precision = 1 : i32{{.*}}b_precision = 1 : i32
// CHECK-NOT: llvm.call
// CHECK-NOT: llvm.func

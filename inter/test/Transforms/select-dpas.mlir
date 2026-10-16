// RUN: inter-opt %s --inter-select-to-machine | FileCheck %s

module {
  func.func @dpas()
      attributes {xemachine.kernel, xemachine.kernel_args = [],
                  xw.simd_width = 16 : i32} {
    %a = xw.constant dense<0> : vector<8xi16> -> !xw.simd<vector<8xi16>, 16>
    %b = xw.constant dense<0> : vector<8xi32> -> !xw.simd<vector<8xi32>, 16>
    %acc = xw.constant dense<0.0> : vector<8xf32> -> !xw.simd<vector<8xf32>, 16>
    %result = xw.dpas %a, %b, %acc {a_precision = 0 : i32, b_precision = 1 : i32, k = 16 : i64, repeat_count = 8 : i64, systolic_depth = 8 : i64} : !xw.simd<vector<8xi16>, 16>, !xw.simd<vector<8xi32>, 16>, !xw.simd<vector<8xf32>, 16> -> !xw.simd<vector<8xf32>, 16>
    return
  }
}

// CHECK-LABEL: func.func @dpas
// CHECK: xemachine.dpas {{.*}}aPrecision = 0 : i32{{.*}}bPrecision = 1 : i32{{.*}}elemType = f32

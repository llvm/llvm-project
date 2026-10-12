// RUN: fir-opt %s | fir-opt | FileCheck %s

// Round-trip the launch configuration attribute a kernel directive records.

// CHECK-LABEL: func.func private @one_dim()
// CHECK-SAME: cuf.launch_config = #cuf.launch_config<dims = 1 : i64, grid = ["*"]>
func.func private @one_dim() attributes {cuf.launch_config = #cuf.launch_config<dims = 1 : i64, grid = ["*"]>} {
  return
}

// CHECK-LABEL: func.func private @two_dims_constant_grid()
// CHECK-SAME: cuf.launch_config = #cuf.launch_config<dims = 2 : i64, grid = ["64", "*"]>
func.func private @two_dims_constant_grid() attributes {cuf.launch_config = #cuf.launch_config<dims = 2 : i64, grid = ["64", "*"]>} {
  return
}

// CHECK-LABEL: func.func private @three_dims_variable_grid()
// CHECK-SAME: cuf.launch_config = #cuf.launch_config<dims = 3 : i64, grid = ["ng", "*", "*"]>
func.func private @three_dims_variable_grid() attributes {cuf.launch_config = #cuf.launch_config<dims = 3 : i64, grid = ["ng", "*", "*"]>} {
  return
}

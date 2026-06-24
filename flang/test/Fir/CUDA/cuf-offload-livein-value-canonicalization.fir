// RUN: fir-opt %s -offload-livein-value-canonicalization -split-input-file | FileCheck %s

// -----

// Test constant sinking into cuf.kernel
func.func @test_constant_sink() {
  %c1 = arith.constant 1 : index
  %c1_i32 = arith.constant 1 : i32
  cuf.kernel<<<%c1_i32, %c1_i32>>> (%arg0 : index) = (%c1 : index) to (%c1 : index) step (%c1 : index) {
    %res = arith.addi %c1, %c1 : index
    "fir.end"() : () -> ()
  }
  return
}

// CHECK-LABEL: @test_constant_sink
// CHECK: cuf.kernel
// CHECK:   arith.constant 1 : index

// -----

// Test constant rematerialization with cuf.kernel
func.func @test_constant_rematerialize() {
  %c1 = arith.constant 1 : index
  %c1_i32 = arith.constant 1 : i32
  %res = arith.addi %c1, %c1 : index
  cuf.kernel<<<%c1_i32, %c1_i32>>> (%arg0 : index) = (%c1 : index) to (%c1 : index) step (%c1 : index) {
    %res2 = arith.addi %c1, %c1 : index
    "fir.end"() : () -> ()
  }
  return
}

// CHECK-LABEL: @test_constant_rematerialize
// CHECK: arith.constant 1 : index
// CHECK: cuf.kernel
// CHECK:   arith.constant 1 : index

// -----

// Test fir.shape sinking into cuf.kernel
func.func @test_firshape_sink() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %shape = fir.shape %c0 : (index) -> !fir.shape<1>
  %c1_i32 = arith.constant 1 : i32
  cuf.kernel<<<%c1_i32, %c1_i32>>> (%arg0 : index) = (%c1 : index) to (%c1 : index) step (%c1 : index) {
    %zeroaddr = fir.zero_bits !fir.heap<!fir.array<?xf32>>
    %box = fir.embox %zeroaddr(%shape) : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xf32>>>
    "fir.end"() : () -> ()
  }
  return
}

// CHECK-LABEL: @test_firshape_sink
// CHECK: cuf.kernel
// CHECK:   fir.shape

// -----

// Test 2D shape sinking with cuf.kernel and array operations
func.func @test_2d_shape_sink() {
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c32 = arith.constant 32 : index
  %0 = cuf.alloc !fir.array<3x32xf32> {bindc_name = "a", data_attr = #cuf.cuda<device>, uniq_name = "_QFEa"} -> !fir.ref<!fir.array<3x32xf32>>
  %shape = fir.shape %c3, %c32 : (index, index) -> !fir.shape<2>
  %decl = fir.declare %0(%shape) {data_attr = #cuf.cuda<device>, uniq_name = "_QFEa"} : (!fir.ref<!fir.array<3x32xf32>>, !fir.shape<2>) -> !fir.ref<!fir.array<3x32xf32>>
  %c1_i32 = arith.constant 1 : i32
  cuf.kernel<<<*, *>>> (%arg0 : index) = (%c1 : index) to (%c32 : index) step (%c1 : index) {
    %coor = fir.array_coor %decl(%shape) %c1, %arg0 : (!fir.ref<!fir.array<3x32xf32>>, !fir.shape<2>, index, index) -> !fir.ref<f32>
    "fir.end"() : () -> ()
  }
  return
}

// CHECK-LABEL: @test_2d_shape_sink
// CHECK: cuf.kernel
// CHECK:   fir.shape

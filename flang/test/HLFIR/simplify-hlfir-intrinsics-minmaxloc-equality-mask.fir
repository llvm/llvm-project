// RUN: fir-opt %s --simplify-hlfir-intrinsics | FileCheck %s

func.func @test_minloc_1d_eq_mask(%input: !fir.box<!fir.array<?xi32>>,
                                   %val: i32) -> !hlfir.expr<1xi32> {
  %c0 = arith.constant 0 : index
  %dims:3 = fir.box_dims %input, %c0 : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
  %shape = fir.shape %dims#1 : (index) -> !fir.shape<1>
  %mask = hlfir.elemental %shape unordered : (!fir.shape<1>) -> !hlfir.expr<?x!fir.logical<4>> {
  ^bb0(%i: index):
    %elem = hlfir.designate %input (%i) : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
    %ld   = fir.load %elem : !fir.ref<i32>
    %cmp  = arith.cmpi eq, %ld, %val : i32
    %conv = fir.convert %cmp : (i1) -> !fir.logical<4>
    hlfir.yield_element %conv : !fir.logical<4>
  }
  %res = hlfir.minloc %input mask %mask {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<?xi32>>, !hlfir.expr<?x!fir.logical<4>>) -> !hlfir.expr<1xi32>
  hlfir.destroy %mask : !hlfir.expr<?x!fir.logical<4>>
  return %res : !hlfir.expr<1xi32>
}
// CHECK-LABEL: func.func @test_minloc_1d_eq_mask(
// CHECK-NOT:     arith.constant 2147483647
// CHECK:         fir.do_loop {{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %true) -> (i32, i1)
// CHECK:           hlfir.apply
// CHECK:           arith.andi
// CHECK:           fir.if {{.*}} -> (i32, i1) {
// CHECK:             fir.result %{{.*}}, %false : i32, i1
// CHECK-NOT:     arith.cmpi slt
// CHECK-NOT:     arith.select {{.*}} : i32

func.func @test_minloc_2d_eq_mask(%input: !fir.box<!fir.array<?x?xi32>>,
                                   %val: i32) -> !hlfir.expr<2xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0:3 = fir.box_dims %input, %c0 : (!fir.box<!fir.array<?x?xi32>>, index) -> (index, index, index)
  %d1:3 = fir.box_dims %input, %c1 : (!fir.box<!fir.array<?x?xi32>>, index) -> (index, index, index)
  %shape = fir.shape %d0#1, %d1#1 : (index, index) -> !fir.shape<2>
  %mask = hlfir.elemental %shape unordered : (!fir.shape<2>) -> !hlfir.expr<?x?x!fir.logical<4>> {
  ^bb0(%i: index, %j: index):
    %elem = hlfir.designate %input (%i, %j) : (!fir.box<!fir.array<?x?xi32>>, index, index) -> !fir.ref<i32>
    %ld   = fir.load %elem : !fir.ref<i32>
    %cmp  = arith.cmpi eq, %ld, %val : i32
    %conv = fir.convert %cmp : (i1) -> !fir.logical<4>
    hlfir.yield_element %conv : !fir.logical<4>
  }
  %res = hlfir.minloc %input mask %mask {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<?x?xi32>>, !hlfir.expr<?x?x!fir.logical<4>>) -> !hlfir.expr<2xi32>
  hlfir.destroy %mask : !hlfir.expr<?x?x!fir.logical<4>>
  return %res : !hlfir.expr<2xi32>
}
// CHECK-LABEL: func.func @test_minloc_2d_eq_mask(
// CHECK-NOT:     arith.constant 2147483647
// CHECK:         fir.do_loop {{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %true) -> (i32, i32, i1)
// CHECK:           fir.do_loop {{.*}} iter_args(%{{.*}}, %{{.*}}, %{{.*}}) -> (i32, i32, i1)
// CHECK:             hlfir.apply
// CHECK:             arith.andi
// CHECK:             fir.if {{.*}} -> (i32, i32, i1) {
// CHECK:               fir.result %{{.*}}, %{{.*}}, %false : i32, i32, i1
// CHECK-NOT:     arith.cmpi slt
// CHECK-NOT:     arith.select {{.*}} : i32

func.func @test_maxloc_1d_eq_mask(%input: !fir.box<!fir.array<?xi32>>,
                                   %val: i32) -> !hlfir.expr<1xi32> {
  %c0 = arith.constant 0 : index
  %dims:3 = fir.box_dims %input, %c0 : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
  %shape = fir.shape %dims#1 : (index) -> !fir.shape<1>
  %mask = hlfir.elemental %shape unordered : (!fir.shape<1>) -> !hlfir.expr<?x!fir.logical<4>> {
  ^bb0(%i: index):
    %elem = hlfir.designate %input (%i) : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
    %ld   = fir.load %elem : !fir.ref<i32>
    %cmp  = arith.cmpi eq, %ld, %val : i32
    %conv = fir.convert %cmp : (i1) -> !fir.logical<4>
    hlfir.yield_element %conv : !fir.logical<4>
  }
  %res = hlfir.maxloc %input mask %mask {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<?xi32>>, !hlfir.expr<?x!fir.logical<4>>) -> !hlfir.expr<1xi32>
  hlfir.destroy %mask : !hlfir.expr<?x!fir.logical<4>>
  return %res : !hlfir.expr<1xi32>
}
// CHECK-LABEL: func.func @test_maxloc_1d_eq_mask(
// CHECK-NOT:     arith.constant -2147483648
// CHECK:         fir.do_loop {{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %true) -> (i32, i1)
// CHECK:           hlfir.apply
// CHECK:           arith.andi
// CHECK:           fir.if {{.*}} -> (i32, i1) {
// CHECK:             fir.result %{{.*}}, %false : i32, i1
// CHECK-NOT:     arith.cmpi sgt

// Partial reduction (DIM=1) with equality mask on 2D array — maxloc.
func.func @test_maxloc_dim1_eq_mask(%input: !fir.box<!fir.array<?x?xi32>>,
                                     %val: i32) -> !hlfir.expr<?xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim = arith.constant 1 : i32
  %d0:3 = fir.box_dims %input, %c0 : (!fir.box<!fir.array<?x?xi32>>, index) -> (index, index, index)
  %d1:3 = fir.box_dims %input, %c1 : (!fir.box<!fir.array<?x?xi32>>, index) -> (index, index, index)
  %shape = fir.shape %d0#1, %d1#1 : (index, index) -> !fir.shape<2>
  %mask = hlfir.elemental %shape unordered : (!fir.shape<2>) -> !hlfir.expr<?x?x!fir.logical<4>> {
  ^bb0(%i: index, %j: index):
    %elem = hlfir.designate %input (%i, %j) : (!fir.box<!fir.array<?x?xi32>>, index, index) -> !fir.ref<i32>
    %ld   = fir.load %elem : !fir.ref<i32>
    %cmp  = arith.cmpi eq, %ld, %val : i32
    %conv = fir.convert %cmp : (i1) -> !fir.logical<4>
    hlfir.yield_element %conv : !fir.logical<4>
  }
  %res = hlfir.maxloc %input dim %dim mask %mask {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<?x?xi32>>, i32, !hlfir.expr<?x?x!fir.logical<4>>) -> !hlfir.expr<?xi32>
  hlfir.destroy %mask : !hlfir.expr<?x?x!fir.logical<4>>
  return %res : !hlfir.expr<?xi32>
}
// CHECK-LABEL: func.func @test_maxloc_dim1_eq_mask(
// CHECK-NOT:     arith.constant -2147483648
// CHECK:         hlfir.elemental
// CHECK:           fir.do_loop {{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %true) -> (i32, i1)
// CHECK:             hlfir.apply
// CHECK:             arith.andi
// CHECK:             fir.if {{.*}} -> (i32, i1) {
// CHECK:               fir.result %{{.*}}, %false : i32, i1
// CHECK-NOT:     arith.cmpi sgt

// hlfir.expr input with equality mask.
func.func @test_minloc_expr_eq_mask(%input: !hlfir.expr<?xi32>,
                                     %val: i32) -> !hlfir.expr<1xi32> {
  %0 = hlfir.shape_of %input : (!hlfir.expr<?xi32>) -> !fir.shape<1>
  %mask = hlfir.elemental %0 unordered : (!fir.shape<1>) -> !hlfir.expr<?x!fir.logical<4>> {
  ^bb0(%i: index):
    %elem = hlfir.apply %input, %i : (!hlfir.expr<?xi32>, index) -> i32
    %cmp  = arith.cmpi eq, %elem, %val : i32
    %conv = fir.convert %cmp : (i1) -> !fir.logical<4>
    hlfir.yield_element %conv : !fir.logical<4>
  }
  %res = hlfir.minloc %input mask %mask {fastmath = #arith.fastmath<contract>} : (!hlfir.expr<?xi32>, !hlfir.expr<?x!fir.logical<4>>) -> !hlfir.expr<1xi32>
  hlfir.destroy %mask : !hlfir.expr<?x!fir.logical<4>>
  return %res : !hlfir.expr<1xi32>
}
// CHECK-LABEL: func.func @test_minloc_expr_eq_mask(
// CHECK-NOT:     arith.constant 2147483647
// CHECK:         fir.do_loop {{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %true) -> (i32, i1)
// CHECK:           hlfir.apply
// CHECK:           arith.andi
// CHECK:           fir.if {{.*}} -> (i32, i1) {
// CHECK:             fir.result %{{.*}}, %false : i32, i1
// CHECK-NOT:     arith.cmpi slt

// 3D total reduction with equality mask.
func.func @test_minloc_3d_eq_mask(%input: !fir.box<!fir.array<?x?x?xi32>>,
                                    %val: i32) -> !hlfir.expr<3xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %d0:3 = fir.box_dims %input, %c0 : (!fir.box<!fir.array<?x?x?xi32>>, index) -> (index, index, index)
  %d1:3 = fir.box_dims %input, %c1 : (!fir.box<!fir.array<?x?x?xi32>>, index) -> (index, index, index)
  %d2:3 = fir.box_dims %input, %c2 : (!fir.box<!fir.array<?x?x?xi32>>, index) -> (index, index, index)
  %shape = fir.shape %d0#1, %d1#1, %d2#1 : (index, index, index) -> !fir.shape<3>
  %mask = hlfir.elemental %shape unordered : (!fir.shape<3>) -> !hlfir.expr<?x?x?x!fir.logical<4>> {
  ^bb0(%i: index, %j: index, %k: index):
    %elem = hlfir.designate %input (%i, %j, %k) : (!fir.box<!fir.array<?x?x?xi32>>, index, index, index) -> !fir.ref<i32>
    %ld   = fir.load %elem : !fir.ref<i32>
    %cmp  = arith.cmpi eq, %ld, %val : i32
    %conv = fir.convert %cmp : (i1) -> !fir.logical<4>
    hlfir.yield_element %conv : !fir.logical<4>
  }
  %res = hlfir.minloc %input mask %mask {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<?x?x?xi32>>, !hlfir.expr<?x?x?x!fir.logical<4>>) -> !hlfir.expr<3xi32>
  hlfir.destroy %mask : !hlfir.expr<?x?x?x!fir.logical<4>>
  return %res : !hlfir.expr<3xi32>
}
// CHECK-LABEL: func.func @test_minloc_3d_eq_mask(
// CHECK-NOT:     arith.constant 2147483647
// CHECK:         fir.do_loop {{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %true) -> (i32, i32, i32, i1)
// CHECK:           fir.do_loop {{.*}} iter_args(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) -> (i32, i32, i32, i1)
// CHECK:             fir.do_loop {{.*}} iter_args(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) -> (i32, i32, i32, i1)
// CHECK:               hlfir.apply
// CHECK:               arith.andi
// CHECK:               fir.if {{.*}} -> (i32, i32, i32, i1) {
// CHECK:                 fir.result %{{.*}}, %{{.*}}, %{{.*}}, %false : i32, i32, i32, i1
// CHECK-NOT:     arith.cmpi slt

// Partial reduction (DIM=1) with equality mask on 2D array.
func.func @test_minloc_dim1_eq_mask(%input: !fir.box<!fir.array<?x?xi32>>,
                                     %val: i32) -> !hlfir.expr<?xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim = arith.constant 1 : i32
  %d0:3 = fir.box_dims %input, %c0 : (!fir.box<!fir.array<?x?xi32>>, index) -> (index, index, index)
  %d1:3 = fir.box_dims %input, %c1 : (!fir.box<!fir.array<?x?xi32>>, index) -> (index, index, index)
  %shape = fir.shape %d0#1, %d1#1 : (index, index) -> !fir.shape<2>
  %mask = hlfir.elemental %shape unordered : (!fir.shape<2>) -> !hlfir.expr<?x?x!fir.logical<4>> {
  ^bb0(%i: index, %j: index):
    %elem = hlfir.designate %input (%i, %j) : (!fir.box<!fir.array<?x?xi32>>, index, index) -> !fir.ref<i32>
    %ld   = fir.load %elem : !fir.ref<i32>
    %cmp  = arith.cmpi eq, %ld, %val : i32
    %conv = fir.convert %cmp : (i1) -> !fir.logical<4>
    hlfir.yield_element %conv : !fir.logical<4>
  }
  %res = hlfir.minloc %input dim %dim mask %mask {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<?x?xi32>>, i32, !hlfir.expr<?x?x!fir.logical<4>>) -> !hlfir.expr<?xi32>
  hlfir.destroy %mask : !hlfir.expr<?x?x!fir.logical<4>>
  return %res : !hlfir.expr<?xi32>
}
// CHECK-LABEL: func.func @test_minloc_dim1_eq_mask(
// CHECK-NOT:     arith.constant 2147483647
// CHECK:         hlfir.elemental
// CHECK:           fir.do_loop {{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %true) -> (i32, i1)
// CHECK:             hlfir.apply
// CHECK:             arith.andi
// CHECK:             fir.if {{.*}} -> (i32, i1) {
// CHECK:               fir.result %{{.*}}, %false : i32, i1
// CHECK-NOT:     arith.cmpi slt

// Partial reduction (DIM=2) with equality mask on 2D array.
func.func @test_minloc_dim2_eq_mask(%input: !fir.box<!fir.array<?x?xi32>>,
                                     %val: i32) -> !hlfir.expr<?xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim = arith.constant 2 : i32
  %d0:3 = fir.box_dims %input, %c0 : (!fir.box<!fir.array<?x?xi32>>, index) -> (index, index, index)
  %d1:3 = fir.box_dims %input, %c1 : (!fir.box<!fir.array<?x?xi32>>, index) -> (index, index, index)
  %shape = fir.shape %d0#1, %d1#1 : (index, index) -> !fir.shape<2>
  %mask = hlfir.elemental %shape unordered : (!fir.shape<2>) -> !hlfir.expr<?x?x!fir.logical<4>> {
  ^bb0(%i: index, %j: index):
    %elem = hlfir.designate %input (%i, %j) : (!fir.box<!fir.array<?x?xi32>>, index, index) -> !fir.ref<i32>
    %ld   = fir.load %elem : !fir.ref<i32>
    %cmp  = arith.cmpi eq, %ld, %val : i32
    %conv = fir.convert %cmp : (i1) -> !fir.logical<4>
    hlfir.yield_element %conv : !fir.logical<4>
  }
  %res = hlfir.minloc %input dim %dim mask %mask {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<?x?xi32>>, i32, !hlfir.expr<?x?x!fir.logical<4>>) -> !hlfir.expr<?xi32>
  hlfir.destroy %mask : !hlfir.expr<?x?x!fir.logical<4>>
  return %res : !hlfir.expr<?xi32>
}
// CHECK-LABEL: func.func @test_minloc_dim2_eq_mask(
// CHECK-NOT:     arith.constant 2147483647
// CHECK:         hlfir.elemental
// CHECK:           fir.do_loop {{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %true) -> (i32, i1)
// CHECK:             hlfir.apply
// CHECK:             arith.andi
// CHECK:             fir.if {{.*}} -> (i32, i1) {
// CHECK:               fir.result %{{.*}}, %false : i32, i1
// CHECK-NOT:     arith.cmpi slt

// 1D expr input, DIM=1, scalar i32 result.
func.func @test_minloc_1d_dim_expr(%input: !hlfir.expr<?xi32>,
                                    %val: i32) -> i32 {
  %dim = arith.constant 1 : i32
  %0 = hlfir.shape_of %input : (!hlfir.expr<?xi32>) -> !fir.shape<1>
  %mask = hlfir.elemental %0 unordered : (!fir.shape<1>) -> !hlfir.expr<?x!fir.logical<4>> {
  ^bb0(%i: index):
    %elem = hlfir.apply %input, %i : (!hlfir.expr<?xi32>, index) -> i32
    %cmp  = arith.cmpi eq, %elem, %val : i32
    %conv = fir.convert %cmp : (i1) -> !fir.logical<4>
    hlfir.yield_element %conv : !fir.logical<4>
  }
  %res = hlfir.minloc %input dim %dim mask %mask {fastmath = #arith.fastmath<contract>} : (!hlfir.expr<?xi32>, i32, !hlfir.expr<?x!fir.logical<4>>) -> i32
  hlfir.destroy %mask : !hlfir.expr<?x!fir.logical<4>>
  return %res : i32
}
// CHECK-LABEL: func.func @test_minloc_1d_dim_expr(
// CHECK-NOT:     arith.constant 2147483647
// CHECK:         fir.do_loop {{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %true) -> (i32, i1)
// CHECK:           hlfir.apply
// CHECK:           arith.andi
// CHECK:           fir.if {{.*}} -> (i32, i1) {
// CHECK:             fir.result %{{.*}}, %false : i32, i1
// CHECK-NOT:     arith.cmpi slt

// 1D box input, i16 result type.
func.func @test_minloc_1d_i16(%input: !fir.box<!fir.array<?xi16>>,
                                %val: i16) -> !hlfir.expr<1xi16> {
  %c0 = arith.constant 0 : index
  %dims:3 = fir.box_dims %input, %c0 : (!fir.box<!fir.array<?xi16>>, index) -> (index, index, index)
  %shape = fir.shape %dims#1 : (index) -> !fir.shape<1>
  %mask = hlfir.elemental %shape unordered : (!fir.shape<1>) -> !hlfir.expr<?x!fir.logical<4>> {
  ^bb0(%i: index):
    %elem = hlfir.designate %input (%i) : (!fir.box<!fir.array<?xi16>>, index) -> !fir.ref<i16>
    %ld   = fir.load %elem : !fir.ref<i16>
    %cmp  = arith.cmpi eq, %ld, %val : i16
    %conv = fir.convert %cmp : (i1) -> !fir.logical<4>
    hlfir.yield_element %conv : !fir.logical<4>
  }
  %res = hlfir.minloc %input mask %mask {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<?xi16>>, !hlfir.expr<?x!fir.logical<4>>) -> !hlfir.expr<1xi16>
  hlfir.destroy %mask : !hlfir.expr<?x!fir.logical<4>>
  return %res : !hlfir.expr<1xi16>
}
// CHECK-LABEL: func.func @test_minloc_1d_i16(
// CHECK-NOT:     arith.constant 32767
// CHECK:         fir.do_loop {{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %true) -> (i16, i1)
// CHECK:           hlfir.apply
// CHECK:           arith.andi
// CHECK:           fir.if {{.*}} -> (i16, i1) {
// CHECK:             fir.result %{{.*}}, %false : i16, i1
// CHECK-NOT:     arith.cmpi slt

// 1D box input, i64 result type.
func.func @test_minloc_1d_i64(%input: !fir.box<!fir.array<?xi64>>,
                                %val: i64) -> !hlfir.expr<1xi64> {
  %c0 = arith.constant 0 : index
  %dims:3 = fir.box_dims %input, %c0 : (!fir.box<!fir.array<?xi64>>, index) -> (index, index, index)
  %shape = fir.shape %dims#1 : (index) -> !fir.shape<1>
  %mask = hlfir.elemental %shape unordered : (!fir.shape<1>) -> !hlfir.expr<?x!fir.logical<4>> {
  ^bb0(%i: index):
    %elem = hlfir.designate %input (%i) : (!fir.box<!fir.array<?xi64>>, index) -> !fir.ref<i64>
    %ld   = fir.load %elem : !fir.ref<i64>
    %cmp  = arith.cmpi eq, %ld, %val : i64
    %conv = fir.convert %cmp : (i1) -> !fir.logical<4>
    hlfir.yield_element %conv : !fir.logical<4>
  }
  %res = hlfir.minloc %input mask %mask {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<?xi64>>, !hlfir.expr<?x!fir.logical<4>>) -> !hlfir.expr<1xi64>
  hlfir.destroy %mask : !hlfir.expr<?x!fir.logical<4>>
  return %res : !hlfir.expr<1xi64>
}
// CHECK-LABEL: func.func @test_minloc_1d_i64(
// CHECK-NOT:     arith.constant 9223372036854775807
// CHECK:         fir.do_loop {{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %true) -> (i64, i1)
// CHECK:           hlfir.apply
// CHECK:           arith.andi
// CHECK:           fir.if {{.*}} -> (i64, i1) {
// CHECK:             fir.result %{{.*}}, %false : i64, i1
// CHECK-NOT:     arith.cmpi slt

// Invariant target on lhs: cmpi eq %val, %ld (swapped operands).
func.func @test_minloc_val_lhs(%input: !fir.box<!fir.array<?xi32>>,
                                %val: i32) -> !hlfir.expr<1xi32> {
  %c0 = arith.constant 0 : index
  %dims:3 = fir.box_dims %input, %c0 : (!fir.box<!fir.array<?xi32>>,
      index) -> (index, index, index)
  %shape = fir.shape %dims#1 : (index) -> !fir.shape<1>
  %mask = hlfir.elemental %shape unordered : (!fir.shape<1>) ->
      !hlfir.expr<?x!fir.logical<4>> {
  ^bb0(%i: index):
    %elem = hlfir.designate %input (%i) : (!fir.box<!fir.array<?xi32>>,
        index) -> !fir.ref<i32>
    %ld   = fir.load %elem : !fir.ref<i32>
    %cmp  = arith.cmpi eq, %val, %ld : i32
    %conv = fir.convert %cmp : (i1) -> !fir.logical<4>
    hlfir.yield_element %conv : !fir.logical<4>
  }
  %res = hlfir.minloc %input mask %mask {fastmath = #arith.fastmath<contract>}
      : (!fir.box<!fir.array<?xi32>>, !hlfir.expr<?x!fir.logical<4>>) ->
        !hlfir.expr<1xi32>
  hlfir.destroy %mask : !hlfir.expr<?x!fir.logical<4>>
  return %res : !hlfir.expr<1xi32>
}
// CHECK-LABEL: func.func @test_minloc_val_lhs(
// CHECK-NOT:     arith.constant 2147483647
// CHECK:         fir.do_loop {{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %true) -> (i32, i1)
// CHECK:           hlfir.apply
// CHECK:           arith.andi
// CHECK:           fir.if {{.*}} -> (i32, i1) {
// CHECK:             fir.result %{{.*}}, %false : i32, i1
// CHECK-NOT:     arith.cmpi slt

// Constant as the invariant target.
func.func @test_minloc_const_target(
    %input: !fir.box<!fir.array<?xi32>>) -> !hlfir.expr<1xi32> {
  %c0 = arith.constant 0 : index
  %c42 = arith.constant 42 : i32
  %dims:3 = fir.box_dims %input, %c0 : (!fir.box<!fir.array<?xi32>>,
      index) -> (index, index, index)
  %shape = fir.shape %dims#1 : (index) -> !fir.shape<1>
  %mask = hlfir.elemental %shape unordered : (!fir.shape<1>) ->
      !hlfir.expr<?x!fir.logical<4>> {
  ^bb0(%i: index):
    %elem = hlfir.designate %input (%i) : (!fir.box<!fir.array<?xi32>>,
        index) -> !fir.ref<i32>
    %ld   = fir.load %elem : !fir.ref<i32>
    %cmp  = arith.cmpi eq, %ld, %c42 : i32
    %conv = fir.convert %cmp : (i1) -> !fir.logical<4>
    hlfir.yield_element %conv : !fir.logical<4>
  }
  %res = hlfir.minloc %input mask %mask {fastmath = #arith.fastmath<contract>}
      : (!fir.box<!fir.array<?xi32>>, !hlfir.expr<?x!fir.logical<4>>) ->
        !hlfir.expr<1xi32>
  hlfir.destroy %mask : !hlfir.expr<?x!fir.logical<4>>
  return %res : !hlfir.expr<1xi32>
}
// CHECK-LABEL: func.func @test_minloc_const_target(
// CHECK-NOT:     arith.constant 2147483647
// CHECK:         fir.do_loop {{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %true) -> (i32, i1)
// CHECK:           hlfir.apply
// CHECK:           arith.andi
// CHECK:           fir.if {{.*}} -> (i32, i1) {
// CHECK:             fir.result %{{.*}}, %false : i32, i1
// CHECK-NOT:     arith.cmpi slt

// Mask elemental wrapped in fir.convert.
func.func @test_minloc_mask_converted(%input: !fir.box<!fir.array<?xi32>>,
                                       %val: i32) -> !hlfir.expr<1xi32> {
  %c0 = arith.constant 0 : index
  %dims:3 = fir.box_dims %input, %c0 : (!fir.box<!fir.array<?xi32>>,
      index) -> (index, index, index)
  %shape = fir.shape %dims#1 : (index) -> !fir.shape<1>
  %mask = hlfir.elemental %shape unordered : (!fir.shape<1>) ->
      !hlfir.expr<?x!fir.logical<4>> {
  ^bb0(%i: index):
    %elem = hlfir.designate %input (%i) : (!fir.box<!fir.array<?xi32>>,
        index) -> !fir.ref<i32>
    %ld   = fir.load %elem : !fir.ref<i32>
    %cmp  = arith.cmpi eq, %ld, %val : i32
    %conv = fir.convert %cmp : (i1) -> !fir.logical<4>
    hlfir.yield_element %conv : !fir.logical<4>
  }
  %wrapped = fir.convert %mask :
      (!hlfir.expr<?x!fir.logical<4>>) -> !hlfir.expr<?x!fir.logical<4>>
  %res = hlfir.minloc %input mask %wrapped {fastmath = #arith.fastmath<contract>}
      : (!fir.box<!fir.array<?xi32>>, !hlfir.expr<?x!fir.logical<4>>) ->
        !hlfir.expr<1xi32>
  hlfir.destroy %mask : !hlfir.expr<?x!fir.logical<4>>
  return %res : !hlfir.expr<1xi32>
}
// CHECK-LABEL: func.func @test_minloc_mask_converted(
// CHECK-NOT:     arith.constant 2147483647
// CHECK:         fir.do_loop {{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %true) -> (i32, i1)
// CHECK:           hlfir.apply
// CHECK:           arith.andi
// CHECK:           fir.if {{.*}} -> (i32, i1) {
// CHECK:             fir.result %{{.*}}, %false : i32, i1
// CHECK-NOT:     arith.cmpi slt

// No mask operand, init from first element.
func.func @test_minloc_no_mask(
    %input: !fir.box<!fir.array<?xi32>>) -> !hlfir.expr<1xi32> {
  %res = hlfir.minloc %input {fastmath = #arith.fastmath<contract>} :
      (!fir.box<!fir.array<?xi32>>) -> !hlfir.expr<1xi32>
  return %res : !hlfir.expr<1xi32>
}
// CHECK-LABEL: func.func @test_minloc_no_mask(
// CHECK-NOT:     arith.constant true
// CHECK:         fir.do_loop {{.*}} iter_args(%{{.*}}, %{{.*}}) -> (i32, i32)
// CHECK:           arith.cmpi slt
// CHECK:           arith.select

// Elemental body yields a constant logical, no comparison op.
func.func @test_minloc_const_logical_mask(
    %input: !fir.box<!fir.array<?xi32>>) -> !hlfir.expr<1xi32> {
  %c0 = arith.constant 0 : index
  %dims:3 = fir.box_dims %input, %c0 : (!fir.box<!fir.array<?xi32>>,
      index) -> (index, index, index)
  %shape = fir.shape %dims#1 : (index) -> !fir.shape<1>
  %mask = hlfir.elemental %shape unordered : (!fir.shape<1>) ->
      !hlfir.expr<?x!fir.logical<4>> {
  ^bb0(%i: index):
    %true = arith.constant true
    %conv = fir.convert %true : (i1) -> !fir.logical<4>
    hlfir.yield_element %conv : !fir.logical<4>
  }
  %res = hlfir.minloc %input mask %mask {fastmath = #arith.fastmath<contract>}
      : (!fir.box<!fir.array<?xi32>>, !hlfir.expr<?x!fir.logical<4>>) ->
        !hlfir.expr<1xi32>
  hlfir.destroy %mask : !hlfir.expr<?x!fir.logical<4>>
  return %res : !hlfir.expr<1xi32>
}
// CHECK-LABEL: func.func @test_minloc_const_logical_mask(
// CHECK:         fir.do_loop {{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %true) -> (i32, i32, i1)
// CHECK:           arith.cmpi slt
// CHECK:           arith.ori
// CHECK:           arith.select

// Following checks for isEqualityMask.

// Mask is a box array, not an hlfir.elemental.
func.func @test_minloc_mask_box(%input: !fir.box<!fir.array<?xi32>>,
    %mask: !fir.box<!fir.array<?x!fir.logical<4>>>) -> !hlfir.expr<1xi32> {
  %res = hlfir.minloc %input mask %mask {fastmath = #arith.fastmath<contract>}
      : (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?x!fir.logical<4>>>) ->
        !hlfir.expr<1xi32>
  return %res : !hlfir.expr<1xi32>
}
// CHECK-LABEL: func.func @test_minloc_mask_box(
// CHECK:         fir.do_loop {{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %true) -> (i32, i32, i1)
// CHECK:           arith.cmpi slt
// CHECK:           arith.ori
// CHECK:           arith.select

// Elemental yields arith.cmpf, not arith.cmpi eq.
func.func @test_minloc_float_mask(%input: !fir.box<!fir.array<?xf32>>,
                                   %val: f32) -> !hlfir.expr<1xi32> {
  %c0 = arith.constant 0 : index
  %dims:3 = fir.box_dims %input, %c0 : (!fir.box<!fir.array<?xf32>>,
      index) -> (index, index, index)
  %shape = fir.shape %dims#1 : (index) -> !fir.shape<1>
  %mask = hlfir.elemental %shape unordered : (!fir.shape<1>) ->
      !hlfir.expr<?x!fir.logical<4>> {
  ^bb0(%i: index):
    %elem = hlfir.designate %input (%i) : (!fir.box<!fir.array<?xf32>>,
        index) -> !fir.ref<f32>
    %ld   = fir.load %elem : !fir.ref<f32>
    %cmp  = arith.cmpf oeq, %ld, %val : f32
    %conv = fir.convert %cmp : (i1) -> !fir.logical<4>
    hlfir.yield_element %conv : !fir.logical<4>
  }
  %res = hlfir.minloc %input mask %mask {fastmath = #arith.fastmath<contract>}
      : (!fir.box<!fir.array<?xf32>>, !hlfir.expr<?x!fir.logical<4>>) ->
        !hlfir.expr<1xi32>
  hlfir.destroy %mask : !hlfir.expr<?x!fir.logical<4>>
  return %res : !hlfir.expr<1xi32>
}
// CHECK-LABEL: func.func @test_minloc_float_mask(
// CHECK:         fir.do_loop {{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %true) -> (i32, f32, i1)
// CHECK:           arith.cmpf olt

// Elemental yields arith.cmpi slt, not eq.
func.func @test_minloc_cmpi_slt_mask(%input: !fir.box<!fir.array<?xi32>>,
                                      %val: i32) -> !hlfir.expr<1xi32> {
  %c0 = arith.constant 0 : index
  %dims:3 = fir.box_dims %input, %c0 : (!fir.box<!fir.array<?xi32>>,
      index) -> (index, index, index)
  %shape = fir.shape %dims#1 : (index) -> !fir.shape<1>
  %mask = hlfir.elemental %shape unordered : (!fir.shape<1>) ->
      !hlfir.expr<?x!fir.logical<4>> {
  ^bb0(%i: index):
    %elem = hlfir.designate %input (%i) : (!fir.box<!fir.array<?xi32>>,
        index) -> !fir.ref<i32>
    %ld   = fir.load %elem : !fir.ref<i32>
    %cmp  = arith.cmpi slt, %ld, %val : i32
    %conv = fir.convert %cmp : (i1) -> !fir.logical<4>
    hlfir.yield_element %conv : !fir.logical<4>
  }
  %res = hlfir.minloc %input mask %mask {fastmath = #arith.fastmath<contract>}
      : (!fir.box<!fir.array<?xi32>>, !hlfir.expr<?x!fir.logical<4>>) ->
        !hlfir.expr<1xi32>
  hlfir.destroy %mask : !hlfir.expr<?x!fir.logical<4>>
  return %res : !hlfir.expr<1xi32>
}
// CHECK-LABEL: func.func @test_minloc_cmpi_slt_mask(
// CHECK:         fir.do_loop {{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %true) -> (i32, i32, i1)
// CHECK:           arith.cmpi slt
// CHECK:           arith.ori
// CHECK:           arith.select

// Mask compares a different array, not the search array.
func.func @test_minloc_different_array_mask(
    %input: !fir.box<!fir.array<?xi32>>,
    %other: !fir.box<!fir.array<?xi32>>,
    %val: i32) -> !hlfir.expr<1xi32> {
  %c0 = arith.constant 0 : index
  %dims:3 = fir.box_dims %other, %c0 : (!fir.box<!fir.array<?xi32>>,
      index) -> (index, index, index)
  %shape = fir.shape %dims#1 : (index) -> !fir.shape<1>
  %mask = hlfir.elemental %shape unordered : (!fir.shape<1>) ->
      !hlfir.expr<?x!fir.logical<4>> {
  ^bb0(%i: index):
    %elem = hlfir.designate %other (%i) : (!fir.box<!fir.array<?xi32>>,
        index) -> !fir.ref<i32>
    %ld   = fir.load %elem : !fir.ref<i32>
    %cmp  = arith.cmpi eq, %ld, %val : i32
    %conv = fir.convert %cmp : (i1) -> !fir.logical<4>
    hlfir.yield_element %conv : !fir.logical<4>
  }
  %res = hlfir.minloc %input mask %mask {fastmath = #arith.fastmath<contract>}
      : (!fir.box<!fir.array<?xi32>>, !hlfir.expr<?x!fir.logical<4>>) ->
        !hlfir.expr<1xi32>
  hlfir.destroy %mask : !hlfir.expr<?x!fir.logical<4>>
  return %res : !hlfir.expr<1xi32>
}
// CHECK-LABEL: func.func @test_minloc_different_array_mask(
// CHECK:         fir.do_loop {{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %true) -> (i32, i32, i1)
// CHECK:           arith.cmpi slt
// CHECK:           arith.ori
// CHECK:           arith.select

// Both cmpi eq operands are loop-invariant scalars.
func.func @test_minloc_both_invariant_mask(
    %input: !fir.box<!fir.array<?xi32>>,
    %a: i32, %b: i32) -> !hlfir.expr<1xi32> {
  %c0 = arith.constant 0 : index
  %dims:3 = fir.box_dims %input, %c0 : (!fir.box<!fir.array<?xi32>>,
      index) -> (index, index, index)
  %shape = fir.shape %dims#1 : (index) -> !fir.shape<1>
  %mask = hlfir.elemental %shape unordered : (!fir.shape<1>) ->
      !hlfir.expr<?x!fir.logical<4>> {
  ^bb0(%i: index):
    %cmp  = arith.cmpi eq, %a, %b : i32
    %conv = fir.convert %cmp : (i1) -> !fir.logical<4>
    hlfir.yield_element %conv : !fir.logical<4>
  }
  %res = hlfir.minloc %input mask %mask {fastmath = #arith.fastmath<contract>}
      : (!fir.box<!fir.array<?xi32>>, !hlfir.expr<?x!fir.logical<4>>) ->
        !hlfir.expr<1xi32>
  hlfir.destroy %mask : !hlfir.expr<?x!fir.logical<4>>
  return %res : !hlfir.expr<1xi32>
}
// CHECK-LABEL: func.func @test_minloc_both_invariant_mask(
// CHECK:         fir.do_loop {{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %true) -> (i32, i32, i1)
// CHECK:           arith.cmpi slt
// CHECK:           arith.ori
// CHECK:           arith.select

// Both cmpi eq operands are variant (two array element loads).
func.func @test_minloc_both_variant_mask(
    %input: !fir.box<!fir.array<?xi32>>,
    %other: !fir.box<!fir.array<?xi32>>) -> !hlfir.expr<1xi32> {
  %c0 = arith.constant 0 : index
  %dims:3 = fir.box_dims %input, %c0 : (!fir.box<!fir.array<?xi32>>,
      index) -> (index, index, index)
  %shape = fir.shape %dims#1 : (index) -> !fir.shape<1>
  %mask = hlfir.elemental %shape unordered : (!fir.shape<1>) ->
      !hlfir.expr<?x!fir.logical<4>> {
  ^bb0(%i: index):
    %e1   = hlfir.designate %input (%i) : (!fir.box<!fir.array<?xi32>>,
        index) -> !fir.ref<i32>
    %ld1  = fir.load %e1 : !fir.ref<i32>
    %e2   = hlfir.designate %other (%i) : (!fir.box<!fir.array<?xi32>>,
        index) -> !fir.ref<i32>
    %ld2  = fir.load %e2 : !fir.ref<i32>
    %cmp  = arith.cmpi eq, %ld1, %ld2 : i32
    %conv = fir.convert %cmp : (i1) -> !fir.logical<4>
    hlfir.yield_element %conv : !fir.logical<4>
  }
  %res = hlfir.minloc %input mask %mask {fastmath = #arith.fastmath<contract>}
      : (!fir.box<!fir.array<?xi32>>, !hlfir.expr<?x!fir.logical<4>>) ->
        !hlfir.expr<1xi32>
  hlfir.destroy %mask : !hlfir.expr<?x!fir.logical<4>>
  return %res : !hlfir.expr<1xi32>
}
// CHECK-LABEL: func.func @test_minloc_both_variant_mask(
// CHECK:         fir.do_loop {{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %true) -> (i32, i32, i1)
// CHECK:           arith.cmpi slt
// CHECK:           arith.ori
// CHECK:           arith.select

// Mask compares a different slice of the same array.
func.func @test_minloc_different_slice_mask(
    %input: !fir.box<!fir.array<?x?xi32>>,
    %val: i32) -> !hlfir.expr<1xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %d0:3 = fir.box_dims %input, %c0 : (!fir.box<!fir.array<?x?xi32>>,
      index) -> (index, index, index)
  %shape = fir.shape %d0#1 : (index) -> !fir.shape<1>
  %col1 = hlfir.designate %input (%c1:%d0#1:%c1, %c1)  shape %shape :
      (!fir.box<!fir.array<?x?xi32>>, index, index, index, index,
       !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
  %col2 = hlfir.designate %input (%c1:%d0#1:%c1, %c2)  shape %shape :
      (!fir.box<!fir.array<?x?xi32>>, index, index, index, index,
       !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
  %mask = hlfir.elemental %shape unordered : (!fir.shape<1>) ->
      !hlfir.expr<?x!fir.logical<4>> {
  ^bb0(%i: index):
    %elem = hlfir.designate %col1 (%i) : (!fir.box<!fir.array<?xi32>>,
        index) -> !fir.ref<i32>
    %ld   = fir.load %elem : !fir.ref<i32>
    %cmp  = arith.cmpi eq, %ld, %val : i32
    %conv = fir.convert %cmp : (i1) -> !fir.logical<4>
    hlfir.yield_element %conv : !fir.logical<4>
  }
  %res = hlfir.minloc %col2 mask %mask {fastmath = #arith.fastmath<contract>}
      : (!fir.box<!fir.array<?xi32>>, !hlfir.expr<?x!fir.logical<4>>) ->
        !hlfir.expr<1xi32>
  hlfir.destroy %mask : !hlfir.expr<?x!fir.logical<4>>
  return %res : !hlfir.expr<1xi32>
}
// CHECK-LABEL: func.func @test_minloc_different_slice_mask(
// CHECK:         fir.do_loop {{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %true) -> (i32, i32, i1)
// CHECK:           arith.cmpi slt
// CHECK:           arith.ori
// CHECK:           arith.select

// Invariant target from a struct field inside the elemental body.
func.func @test_minloc_struct_field_target(
    %input: !fir.box<!fir.array<?xi32>>,
    %config: !fir.ref<!fir.type<cfg{threshold:i32}>>) -> !hlfir.expr<1xi32> {
  %c0 = arith.constant 0 : index
  %dims:3 = fir.box_dims %input, %c0 : (!fir.box<!fir.array<?xi32>>,
      index) -> (index, index, index)
  %shape = fir.shape %dims#1 : (index) -> !fir.shape<1>
  %mask = hlfir.elemental %shape unordered : (!fir.shape<1>) ->
      !hlfir.expr<?x!fir.logical<4>> {
  ^bb0(%i: index):
    %fld  = hlfir.designate %config {"threshold"} :
        (!fir.ref<!fir.type<cfg{threshold:i32}>>) -> !fir.ref<i32>
    %tval = fir.load %fld : !fir.ref<i32>
    %elem = hlfir.designate %input (%i) : (!fir.box<!fir.array<?xi32>>,
        index) -> !fir.ref<i32>
    %ld   = fir.load %elem : !fir.ref<i32>
    %cmp  = arith.cmpi eq, %ld, %tval : i32
    %conv = fir.convert %cmp : (i1) -> !fir.logical<4>
    hlfir.yield_element %conv : !fir.logical<4>
  }
  %res = hlfir.minloc %input mask %mask {fastmath = #arith.fastmath<contract>}
      : (!fir.box<!fir.array<?xi32>>, !hlfir.expr<?x!fir.logical<4>>) ->
        !hlfir.expr<1xi32>
  hlfir.destroy %mask : !hlfir.expr<?x!fir.logical<4>>
  return %res : !hlfir.expr<1xi32>
}
// CHECK-LABEL: func.func @test_minloc_struct_field_target(
// CHECK-NOT:     arith.constant 2147483647
// CHECK:         fir.do_loop {{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %true) -> (i32, i1)
// CHECK:           hlfir.apply
// CHECK:           arith.andi
// CHECK:           fir.if {{.*}} -> (i32, i1) {
// CHECK:             fir.result %{{.*}}, %false : i32, i1
// CHECK-NOT:     arith.cmpi slt

// Invariant target from a constant-indexed global array element.
func.func @test_minloc_global_elem_target(
    %input: !fir.box<!fir.array<?xi32>>,
    %globals: !fir.box<!fir.array<4xi32>>) -> !hlfir.expr<1xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dims:3 = fir.box_dims %input, %c0 : (!fir.box<!fir.array<?xi32>>,
      index) -> (index, index, index)
  %shape = fir.shape %dims#1 : (index) -> !fir.shape<1>
  %mask = hlfir.elemental %shape unordered : (!fir.shape<1>) ->
      !hlfir.expr<?x!fir.logical<4>> {
  ^bb0(%i: index):
    %gptr = hlfir.designate %globals (%c1) : (!fir.box<!fir.array<4xi32>>,
        index) -> !fir.ref<i32>
    %gval = fir.load %gptr : !fir.ref<i32>
    %elem = hlfir.designate %input (%i) : (!fir.box<!fir.array<?xi32>>,
        index) -> !fir.ref<i32>
    %ld   = fir.load %elem : !fir.ref<i32>
    %cmp  = arith.cmpi eq, %ld, %gval : i32
    %conv = fir.convert %cmp : (i1) -> !fir.logical<4>
    hlfir.yield_element %conv : !fir.logical<4>
  }
  %res = hlfir.minloc %input mask %mask {fastmath = #arith.fastmath<contract>}
      : (!fir.box<!fir.array<?xi32>>, !hlfir.expr<?x!fir.logical<4>>) ->
        !hlfir.expr<1xi32>
  hlfir.destroy %mask : !hlfir.expr<?x!fir.logical<4>>
  return %res : !hlfir.expr<1xi32>
}
// CHECK-LABEL: func.func @test_minloc_global_elem_target(
// CHECK-NOT:     arith.constant 2147483647
// CHECK:         fir.do_loop {{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %true) -> (i32, i1)
// CHECK:           hlfir.apply
// CHECK:           arith.andi
// CHECK:           fir.if {{.*}} -> (i32, i1) {
// CHECK:             fir.result %{{.*}}, %false : i32, i1
// CHECK-NOT:     arith.cmpi slt

// Mask uses A(%i+1) — shifted index is variant, optimization must not happen.
func.func @test_minloc_shifted_index_mask(
    %input: !fir.box<!fir.array<?xi32>>) -> !hlfir.expr<1xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dims:3 = fir.box_dims %input, %c0 : (!fir.box<!fir.array<?xi32>>,
      index) -> (index, index, index)
  %shape = fir.shape %dims#1 : (index) -> !fir.shape<1>
  %mask = hlfir.elemental %shape unordered : (!fir.shape<1>) ->
      !hlfir.expr<?x!fir.logical<4>> {
  ^bb0(%i: index):
    %ip1  = arith.addi %i, %c1 : index
    %nptr = hlfir.designate %input (%ip1) : (!fir.box<!fir.array<?xi32>>,
        index) -> !fir.ref<i32>
    %nval = fir.load %nptr : !fir.ref<i32>
    %elem = hlfir.designate %input (%i) : (!fir.box<!fir.array<?xi32>>,
        index) -> !fir.ref<i32>
    %ld   = fir.load %elem : !fir.ref<i32>
    %cmp  = arith.cmpi eq, %ld, %nval : i32
    %conv = fir.convert %cmp : (i1) -> !fir.logical<4>
    hlfir.yield_element %conv : !fir.logical<4>
  }
  %res = hlfir.minloc %input mask %mask {fastmath = #arith.fastmath<contract>}
      : (!fir.box<!fir.array<?xi32>>, !hlfir.expr<?x!fir.logical<4>>) ->
        !hlfir.expr<1xi32>
  hlfir.destroy %mask : !hlfir.expr<?x!fir.logical<4>>
  return %res : !hlfir.expr<1xi32>
}
// CHECK-LABEL: func.func @test_minloc_shifted_index_mask(
// CHECK:         fir.do_loop {{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %true) -> (i32, i32, i1)
// CHECK:           arith.cmpi slt
// CHECK:           arith.ori
// CHECK:           arith.select

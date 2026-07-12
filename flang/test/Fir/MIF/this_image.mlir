// RUN: fir-opt --mif-convert %s | FileCheck %s

  func.func @_QQmain() attributes {fir.bindc_name = "TEST"} {
    %0 = fir.alloca !fir.array<1xi64>
    %1 = fir.alloca !fir.array<2xi64>
    %2 = fir.dummy_scope : !fir.dscope
    %3 = fir.address_of(@_QFEa) : !fir.ref<i32>
    %c1_i64 = arith.constant 1 : i64
    %c0 = arith.constant 0 : index
    %4 = fir.coordinate_of %1, %c0 : (!fir.ref<!fir.array<2xi64>>, index) -> !fir.ref<i64>
    fir.store %c1_i64 to %4 : !fir.ref<i64>
    %c1_i64_0 = arith.constant 1 : i64
    %c1 = arith.constant 1 : index
    %5 = fir.coordinate_of %1, %c1 : (!fir.ref<!fir.array<2xi64>>, index) -> !fir.ref<i64>
    fir.store %c1_i64_0 to %5 : !fir.ref<i64>
    %6 = fir.embox %1 : (!fir.ref<!fir.array<2xi64>>) -> !fir.box<!fir.array<2xi64>>
    %c1_i64_1 = arith.constant 1 : i64
    %c0_2 = arith.constant 0 : index
    %7 = fir.coordinate_of %0, %c0_2 : (!fir.ref<!fir.array<1xi64>>, index) -> !fir.ref<i64>
    fir.store %c1_i64_1 to %7 : !fir.ref<i64>
    %8 = fir.embox %0 : (!fir.ref<!fir.array<1xi64>>) -> !fir.box<!fir.array<1xi64>>
    mif.alloc_coarray %3 lcobounds %6 ucobounds %8 {uniq_name = "_QFEa"} : (!fir.ref<i32>, !fir.box<!fir.array<2xi64>>, !fir.box<!fir.array<1xi64>>) -> ()
    %9:2 = hlfir.declare %3 {uniq_name = "_QFEa"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %10 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFEi"}
    %11:2 = hlfir.declare %10 {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %c2 = arith.constant 2 : index
    %12 = fir.alloca !fir.array<2xi32> {bindc_name = "j", uniq_name = "_QFEj"}
    %13 = fir.shape %c2 : (index) -> !fir.shape<1>
    %14:2 = hlfir.declare %12(%13) {uniq_name = "_QFEj"} : (!fir.ref<!fir.array<2xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<2xi32>>, !fir.ref<!fir.array<2xi32>>)
    %15 = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}> {bindc_name = "team", uniq_name = "_QFEteam"}
    %16:2 = hlfir.declare %15 {uniq_name = "_QFEteam"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>) -> (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>)
    %17 = fir.address_of(@_QQ_QM__fortran_builtinsT__builtin_team_type.DerivedInit) : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>
    fir.copy %17 to %16#0 no_overlap : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>
    %18 = mif.this_image : () -> i32
    hlfir.assign %18 to %11#0 : i32, !fir.ref<i32>
    %19 = fir.embox %16#0 : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>) -> !fir.box<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>
    %20 = mif.this_image team %19 : (!fir.box<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>) -> i32
    hlfir.assign %20 to %11#0 : i32, !fir.ref<i32>
    %21 = fir.embox %9#0 : (!fir.ref<i32>) -> !fir.box<i32, corank:2>
    %22 = mif.this_image coarray %21 : (!fir.box<i32, corank:2>) -> !fir.box<!fir.array<?xi64>>
    %23:2 = hlfir.declare %22 {uniq_name = ".tmp.intrinsic_result"} : (!fir.box<!fir.array<?xi64>>) -> (!fir.box<!fir.array<?xi64>>, !fir.box<!fir.array<?xi64>>)
    %false = arith.constant false
    %24 = hlfir.as_expr %23#0 move %false : (!fir.box<!fir.array<?xi64>>, i1) -> !hlfir.expr<?xi64>
    %c0_3 = arith.constant 0 : index
    %25:3 = fir.box_dims %23#0, %c0_3 : (!fir.box<!fir.array<?xi64>>, index) -> (index, index, index)
    %26 = fir.shape %25#1 : (index) -> !fir.shape<1>
    %27 = hlfir.elemental %26 unordered : (!fir.shape<1>) -> !hlfir.expr<?xi32> {
    ^bb0(%arg0: index):
      %31 = hlfir.apply %24, %arg0 : (!hlfir.expr<?xi64>, index) -> i64
      %32 = fir.convert %31 : (i64) -> i32
      hlfir.yield_element %32 : i32
    }
    hlfir.assign %27 to %14#0 : !hlfir.expr<?xi32>, !fir.ref<!fir.array<2xi32>>
    hlfir.destroy %27 : !hlfir.expr<?xi32>
    hlfir.destroy %24 : !hlfir.expr<?xi64>
    %c1_i32 = arith.constant 1 : i32
    %28 = fir.embox %9#0 : (!fir.ref<i32>) -> !fir.box<i32, corank:2>
    %29 = mif.this_image coarray %28 dim %c1_i32 : (!fir.box<i32, corank:2>, i32) -> i64
    %30 = fir.convert %29 : (i64) -> i32
    hlfir.assign %30 to %14#0 : i32, !fir.ref<!fir.array<2xi32>>
    return
  }

  // Test that MIFThisImageOpConversion emits an element-wise i64→i32
  // conversion loop when the declared element type differs from the i64
  // written by prif_this_image_with_coarray. Models `integer :: a[2,*]`
  // (corank:2, default kind = i32).
  func.func @test_this_image_i32() {
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %lc = fir.alloca !fir.array<2xi64>
    %uc = fir.alloca !fir.array<1xi64>
    %lc0 = fir.coordinate_of %lc, %c0 : (!fir.ref<!fir.array<2xi64>>, index) -> !fir.ref<i64>
    fir.store %c1_i64 to %lc0 : !fir.ref<i64>
    %lc1 = fir.coordinate_of %lc, %c1 : (!fir.ref<!fir.array<2xi64>>, index) -> !fir.ref<i64>
    fir.store %c1_i64 to %lc1 : !fir.ref<i64>
    %uc0 = fir.coordinate_of %uc, %c0 : (!fir.ref<!fir.array<1xi64>>, index) -> !fir.ref<i64>
    fir.store %c2_i64 to %uc0 : !fir.ref<i64>
    %lb = fir.embox %lc : (!fir.ref<!fir.array<2xi64>>) -> !fir.box<!fir.array<2xi64>>
    %ub = fir.embox %uc : (!fir.ref<!fir.array<1xi64>>) -> !fir.box<!fir.array<1xi64>>
    %a_ref = fir.address_of(@_QFEa2) : !fir.ref<i32>
    mif.alloc_coarray %a_ref lcobounds %lb ucobounds %ub {uniq_name = "_QFEa2"} : (!fir.ref<i32>, !fir.box<!fir.array<2xi64>>, !fir.box<!fir.array<1xi64>>) -> ()
    %a_decl:2 = hlfir.declare %a_ref {uniq_name = "_QFEa2"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %cobox = fir.embox %a_decl#0 : (!fir.ref<i32>) -> !fir.box<i32, corank:2>
    %res = mif.this_image coarray %cobox : (!fir.box<i32, corank:2>) -> !fir.box<!fir.array<?xi32>>
    return
  }

// CHECK-LABEL: func.func @_QQmain
// CHECK: fir.call @_QMprifPprif_this_image_no_coarray
// CHECK: fir.call @_QMprifPprif_this_image_with_coarray
// CHECK: fir.call @_QMprifPprif_this_image_with_dim

// CHECK-LABEL: func.func @test_this_image_i32
// i32 and i64 scratch buffers are hoisted to the function entry block.
// CHECK-DAG:     %[[I32BUF:.*]] = fir.alloca !fir.array<2xi32>
// CHECK-DAG:     %[[I64BUF:.*]] = fir.alloca !fir.array<2xi64>
// prif_this_image_with_coarray writes i64 values into the i64 buffer.
// CHECK:         fir.call @_QMprifPprif_this_image_with_coarray
// After the call the i64 box is widened and its base pointer extracted.
// CHECK:         %[[I64BOX:.*]] = fir.convert {{.*}} : (!fir.box<!fir.array<2xi64>>) -> !fir.box<!fir.array<?xi64>>
// CHECK:         %[[I64BUFREF:.*]] = fir.box_addr %[[I64BOX]] : (!fir.box<!fir.array<?xi64>>) -> !fir.ref<!fir.array<?xi64>>
// Loop iteration 0: load i64, truncate to i32, store into i32 buffer.
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[SRC0:.*]] = fir.coordinate_of %[[I64BUFREF]], %[[C0]] : (!fir.ref<!fir.array<?xi64>>, index) -> !fir.ref<i64>
// CHECK:         %[[VAL0:.*]] = fir.load %[[SRC0]] : !fir.ref<i64>
// CHECK:         %[[CONV0:.*]] = fir.convert %[[VAL0]] : (i64) -> i32
// CHECK:         %[[DST0:.*]] = fir.coordinate_of %[[I32BUF]], %[[C0]] : (!fir.ref<!fir.array<2xi32>>, index) -> !fir.ref<i32>
// CHECK:         fir.store %[[CONV0]] to %[[DST0]] : !fir.ref<i32>
// Loop iteration 1: same pattern for the second codimension.
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[SRC1:.*]] = fir.coordinate_of %[[I64BUFREF]], %[[C1]] : (!fir.ref<!fir.array<?xi64>>, index) -> !fir.ref<i64>
// CHECK:         %[[VAL1:.*]] = fir.load %[[SRC1]] : !fir.ref<i64>
// CHECK:         %[[CONV1:.*]] = fir.convert %[[VAL1]] : (i64) -> i32
// CHECK:         %[[DST1:.*]] = fir.coordinate_of %[[I32BUF]], %[[C1]] : (!fir.ref<!fir.array<2xi32>>, index) -> !fir.ref<i32>
// CHECK:         fir.store %[[CONV1]] to %[[DST1]] : !fir.ref<i32>
// The i32 buffer is boxed and widened to a dynamic-extent result.
// CHECK:         %[[BOX_I32:.*]] = fir.embox %[[I32BUF]] : (!fir.ref<!fir.array<2xi32>>) -> !fir.box<!fir.array<2xi32>>
// CHECK:         fir.convert %[[BOX_I32]] : (!fir.box<!fir.array<2xi32>>) -> !fir.box<!fir.array<?xi32>>

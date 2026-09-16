// Test inlining of hlfir.assign for self-copy with scalar index and triplet dimensions.
// RUN: fir-opt --inline-hlfir-assign %s | FileCheck %s

// Fortran source code:
//  subroutine copy_row(a, n, i, j, lo1, hi1, lo2, hi2)
//    integer, intent(in) :: n, i, j, lo1, hi1, lo2, hi2
//    real, intent(inout) :: a(n, n)
//    a(i, lo1:hi1) = a(j, lo2:hi2)
//  end subroutine copy_row

// CHECK-LABEL: func.func @_QPcopy_row

// Index-based disjointness check for scalar index (dim1) and triplet (dim2):
// Dim1 (scalar): compares i vs j : disjoint if i != j (i.e., i < j || j < i)
// CHECK: %[[CMP1:.*]] = arith.cmpi slt, %{{.*}}, %{{.*}} : index
// CHECK: %[[CMP2:.*]] = arith.cmpi slt, %{{.*}}, %{{.*}} : index
// CHECK: %[[DISJ_DIM1:.*]] = arith.ori %[[CMP1]], %[[CMP2]] : i1

// Dim2 (triplet): compares lo1:hi1 vs lo2:hi2 — disjoint if hi1 < lo2 || hi2 < lo1
// CHECK: %[[CMP3:.*]] = arith.cmpi slt, %{{.*}}, %{{.*}} : index
// CHECK: %[[CMP4:.*]] = arith.cmpi slt, %{{.*}}, %{{.*}} : index
// CHECK: %[[DISJ_DIM2:.*]] = arith.ori %[[CMP3]], %[[CMP4]] : i1
// Overall: disjoint if ANY dimension is disjoint
// CHECK: %[[DISJOINT:.*]] = arith.ori %[[DISJ_DIM1]], %[[DISJ_DIM2]] : i1
// CHECK: fir.if %[[DISJOINT]] {
// CHECK:   fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} unordered {
// CHECK:     hlfir.designate
// CHECK:     fir.load
// CHECK:     hlfir.designate
// CHECK:     hlfir.assign %{{.*}} to %{{.*}} : f32, !fir.ref<f32>
// CHECK:   }
// CHECK: } else {
// CHECK:   %[[ALLOC:.*]] = fir.allocmem !fir.array<?xf32>, %{{.*}} {bindc_name = ".tmp", uniq_name = ""}
// CHECK:   %[[TEMP:.*]]:2 = hlfir.declare %[[ALLOC]](%{{.*}}) {uniq_name = ".tmp"}
// CHECK:   fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} unordered {
// CHECK:     hlfir.designate
// CHECK:     fir.load
// CHECK:     hlfir.designate %[[TEMP]]#0
// CHECK:     hlfir.assign %{{.*}} to %{{.*}} temporary_lhs : f32, !fir.ref<f32>
// CHECK:   }
// CHECK:   fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} unordered {
// CHECK:     hlfir.designate %[[TEMP]]#0
// CHECK:     fir.load
// CHECK:     hlfir.designate
// CHECK:     hlfir.assign %{{.*}} to %{{.*}} : f32, !fir.ref<f32>
// CHECK:   }
// CHECK:   fir.freemem %{{.*}}
// CHECK: }

func.func @_QPcopy_row(%arg0: !fir.ref<!fir.array<?x?xf32>> {fir.bindc_name = "a"}, %arg1: !fir.ref<i32> {fir.bindc_name = "n"}, %arg2: !fir.ref<i32> {fir.bindc_name = "i"}, %arg3: !fir.ref<i32> {fir.bindc_name = "j"}, %arg4: !fir.ref<i32> {fir.bindc_name = "lo1"}, %arg5: !fir.ref<i32> {fir.bindc_name = "hi1"}, %arg6: !fir.ref<i32> {fir.bindc_name = "lo2"}, %arg7: !fir.ref<i32> {fir.bindc_name = "hi2"}) {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg1 dummy_scope %0 arg 2 {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFcopy_rowEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %2:2 = hlfir.declare %arg2 dummy_scope %0 arg 3 {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFcopy_rowEi"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %3:2 = hlfir.declare %arg3 dummy_scope %0 arg 4 {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFcopy_rowEj"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %4:2 = hlfir.declare %arg4 dummy_scope %0 arg 5 {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFcopy_rowElo1"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %5:2 = hlfir.declare %arg5 dummy_scope %0 arg 6 {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFcopy_rowEhi1"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %6:2 = hlfir.declare %arg6 dummy_scope %0 arg 7 {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFcopy_rowElo2"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %7:2 = hlfir.declare %arg7 dummy_scope %0 arg 8 {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFcopy_rowEhi2"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)

  %8 = fir.load %1#0 : !fir.ref<i32>
  %9 = fir.convert %8 : (i32) -> i64
  %10 = fir.convert %9 : (i64) -> index
  %11 = arith.cmpi sgt, %10, %c0 : index
  %12 = arith.select %11, %10, %c0 : index
  %13 = fir.shape %12, %12 : (index, index) -> !fir.shape<2>
  %14:2 = hlfir.declare %arg0(%13) dummy_scope %0 arg 1 {fortran_attrs = #fir.var_attrs<intent_inout>, uniq_name = "_QFcopy_rowEa"} : (!fir.ref<!fir.array<?x?xf32>>, !fir.shape<2>, !fir.dscope) -> (!fir.box<!fir.array<?x?xf32>>, !fir.ref<!fir.array<?x?xf32>>)

  // RHS: a(j, lo2:hi2)
  %15 = fir.load %3#0 : !fir.ref<i32>
  %16 = fir.convert %15 : (i32) -> i64
  %17 = fir.convert %16 : (i64) -> index
  %18 = fir.load %6#0 : !fir.ref<i32>
  %19 = fir.convert %18 : (i32) -> i64
  %20 = fir.convert %19 : (i64) -> index
  %21 = fir.load %7#0 : !fir.ref<i32>
  %22 = fir.convert %21 : (i32) -> i64
  %23 = fir.convert %22 : (i64) -> index
  %24 = arith.subi %23, %20 : index
  %25 = arith.addi %24, %c1 : index
  %26 = arith.cmpi sgt, %25, %c0 : index
  %27 = arith.select %26, %25, %c0 : index
  %28 = fir.shape %27 : (index) -> !fir.shape<1>
  %29 = hlfir.designate %14#0 (%17, %20:%23:%c1)  shape %28 : (!fir.box<!fir.array<?x?xf32>>, index, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>

  // LHS: a(i, lo1:hi1)
  %30 = fir.load %2#0 : !fir.ref<i32>
  %31 = fir.convert %30 : (i32) -> i64
  %32 = fir.convert %31 : (i64) -> index
  %33 = fir.load %4#0 : !fir.ref<i32>
  %34 = fir.convert %33 : (i32) -> i64
  %35 = fir.convert %34 : (i64) -> index
  %36 = fir.load %5#0 : !fir.ref<i32>
  %37 = fir.convert %36 : (i32) -> i64
  %38 = fir.convert %37 : (i64) -> index
  %39 = arith.subi %38, %35 : index
  %40 = arith.addi %39, %c1 : index
  %41 = arith.cmpi sgt, %40, %c0 : index
  %42 = arith.select %41, %40, %c0 : index
  %43 = fir.shape %42 : (index) -> !fir.shape<1>
  %44 = hlfir.designate %14#0 (%32, %35:%38:%c1)  shape %43 : (!fir.box<!fir.array<?x?xf32>>, index, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>

  hlfir.assign %29 to %44 : !fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>
  return
}

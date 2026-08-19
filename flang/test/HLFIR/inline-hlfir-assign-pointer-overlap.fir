// Test inlining of hlfir.assign for pointer-based array assignment

// RUN: fir-opt --inline-hlfir-assign %s | FileCheck %s

// Fortran source code:
//  subroutine assign_pointers(arr, N, lo1, hi1, lo2, hi2)
//    integer, intent(in)    :: N, lo1, hi1, lo2, hi2
//    integer, pointer, intent(inout) :: arr(:)
//    integer, pointer :: p1(:), p2(:)
//
//    p1 => arr(lo1:hi1)
//    p2 => arr(lo2:hi2)
//
//    p1 = p2
//  end subroutine assign_pointers

// CHECK-LABEL: func.func @_QPassign_pointers

// Address-based runtime disjointness check:
// CHECK: %[[LHS_ADDR:.*]] = fir.box_addr %{{.*}} : (!fir.box<!fir.ptr<!fir.array<?xi32>>>) -> !fir.ptr<!fir.array<?xi32>>
// CHECK: %[[LHS_BASE:.*]] = fir.convert %[[LHS_ADDR]] : (!fir.ptr<!fir.array<?xi32>>) -> i64
// CHECK: fir.box_elesize %{{.*}} : (!fir.box<!fir.ptr<!fir.array<?xi32>>>) -> index
// CHECK: fir.box_dims %{{.*}}, %{{.*}}
// CHECK: arith.subi %{{.*}}, %{{.*}} : index
// CHECK: arith.muli %{{.*}}, %{{.*}} : index
// CHECK: arith.cmpi slt, %{{.*}}, %{{.*}} : index
// CHECK: arith.select %{{.*}}, %{{.*}}, %{{.*}} : index
// CHECK: arith.select %{{.*}}, %{{.*}}, %{{.*}} : index
// CHECK: arith.subi %{{.*}}, %{{.*}} : index
// CHECK: arith.addi %{{.*}}, %{{.*}} : index
// CHECK: fir.convert %{{.*}} : (index) -> i64
// CHECK: fir.convert %{{.*}} : (index) -> i64
// CHECK: %[[LHS_START:.*]] = arith.addi %[[LHS_BASE]], %{{.*}} : i64
// CHECK: %[[LHS_END:.*]] = arith.addi %[[LHS_BASE]], %{{.*}} : i64
// CHECK: %[[RHS_ADDR:.*]] = fir.box_addr %{{.*}} : (!fir.box<!fir.ptr<!fir.array<?xi32>>>) -> !fir.ptr<!fir.array<?xi32>>
// CHECK: %[[RHS_BASE:.*]] = fir.convert %[[RHS_ADDR]] : (!fir.ptr<!fir.array<?xi32>>) -> i64
// CHECK: fir.box_elesize %{{.*}} : (!fir.box<!fir.ptr<!fir.array<?xi32>>>) -> index
// CHECK: fir.box_dims %{{.*}}, %{{.*}}
// CHECK: arith.subi %{{.*}}, %{{.*}} : index
// CHECK: arith.muli %{{.*}}, %{{.*}} : index
// CHECK: arith.cmpi slt, %{{.*}}, %{{.*}} : index
// CHECK: arith.select %{{.*}}, %{{.*}}, %{{.*}} : index
// CHECK: arith.select %{{.*}}, %{{.*}}, %{{.*}} : index
// CHECK: arith.subi %{{.*}}, %{{.*}} : index
// CHECK: arith.addi %{{.*}}, %{{.*}} : index
// CHECK: fir.convert %{{.*}} : (index) -> i64
// CHECK: fir.convert %{{.*}} : (index) -> i64
// CHECK: %[[RHS_START:.*]] = arith.addi %[[RHS_BASE]], %{{.*}} : i64
// CHECK: %[[RHS_END:.*]] = arith.addi %[[RHS_BASE]], %{{.*}} : i64
// CHECK: %[[CMP1:.*]] = arith.cmpi ult, %[[LHS_END]], %[[RHS_START]] : i64
// CHECK: %[[CMP2:.*]] = arith.cmpi ult, %[[RHS_END]], %[[LHS_START]] : i64
// CHECK: %[[DISJOINT:.*]] = arith.ori %[[CMP1]], %[[CMP2]] : i1
// CHECK: fir.if %[[DISJOINT]] {
// CHECK:   fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} unordered {
// CHECK:     hlfir.designate
// CHECK:     fir.load
// CHECK:     hlfir.designate
// CHECK:     hlfir.assign %{{.*}} to %{{.*}} : i32, !fir.ref<i32>
// CHECK:   }
// CHECK: } else {
// CHECK:   %[[ALLOC:.*]] = fir.allocmem !fir.array<?xi32>, %{{.*}} {bindc_name = ".tmp", uniq_name = ""}
// CHECK:   %[[TEMP:.*]]:2 = hlfir.declare %[[ALLOC]](%{{.*}}) {uniq_name = ".tmp"}
// CHECK:   fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} unordered {
// CHECK:     hlfir.designate
// CHECK:     fir.load
// CHECK:     hlfir.designate %[[TEMP]]#0
// CHECK:     hlfir.assign %{{.*}} to %{{.*}} temporary_lhs : i32, !fir.ref<i32>
// CHECK:   }
// CHECK:   fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} unordered {
// CHECK:     hlfir.designate %[[TEMP]]#0
// CHECK:     fir.load
// CHECK:     hlfir.designate
// CHECK:     hlfir.assign %{{.*}} to %{{.*}} : i32, !fir.ref<i32>
// CHECK:   }
// CHECK:   fir.freemem %{{.*}}
// CHECK: }

func.func @_QPassign_pointers(%arg0: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>> {fir.bindc_name = "arr"}, %arg1: !fir.ref<i32> {fir.bindc_name = "n"}, %arg2: !fir.ref<i32> {fir.bindc_name = "lo1"}, %arg3: !fir.ref<i32> {fir.bindc_name = "hi1"}, %arg4: !fir.ref<i32> {fir.bindc_name = "lo2"}, %arg5: !fir.ref<i32> {fir.bindc_name = "hi2"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg0 dummy_scope %0 arg 1 {fortran_attrs = #fir.var_attrs<intent_inout, pointer>, uniq_name = "_QFassign_pointersEarr"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>)
  %2:2 = hlfir.declare %arg3 dummy_scope %0 arg 4 {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFassign_pointersEhi1"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %3:2 = hlfir.declare %arg5 dummy_scope %0 arg 6 {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFassign_pointersEhi2"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %4:2 = hlfir.declare %arg2 dummy_scope %0 arg 3 {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFassign_pointersElo1"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %5:2 = hlfir.declare %arg4 dummy_scope %0 arg 5 {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFassign_pointersElo2"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %6:2 = hlfir.declare %arg1 dummy_scope %0 arg 2 {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFassign_pointersEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %7 = fir.alloca !fir.box<!fir.ptr<!fir.array<?xi32>>> {bindc_name = "p1", uniq_name = "_QFassign_pointersEp1"}
  %8 = fir.zero_bits !fir.ptr<!fir.array<?xi32>>
  %c0 = arith.constant 0 : index
  %9 = fir.shape %c0 : (index) -> !fir.shape<1>
  %10 = fir.embox %8(%9) : (!fir.ptr<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>>
  fir.store %10 to %7 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
  %11:2 = hlfir.declare %7 {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFassign_pointersEp1"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>)
  %12 = fir.alloca !fir.box<!fir.ptr<!fir.array<?xi32>>> {bindc_name = "p2", uniq_name = "_QFassign_pointersEp2"}
  %13 = fir.zero_bits !fir.ptr<!fir.array<?xi32>>
  %c0_0 = arith.constant 0 : index
  %14 = fir.shape %c0_0 : (index) -> !fir.shape<1>
  %15 = fir.embox %13(%14) : (!fir.ptr<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>>
  fir.store %15 to %12 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
  %16:2 = hlfir.declare %12 {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFassign_pointersEp2"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>)
  %17 = fir.load %1#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
  %18 = fir.load %4#0 : !fir.ref<i32>
  %19 = fir.convert %18 : (i32) -> i64
  %20 = fir.load %2#0 : !fir.ref<i32>
  %21 = fir.convert %20 : (i32) -> i64
  %22 = fir.convert %19 : (i64) -> index
  %23 = fir.convert %21 : (i64) -> index
  %c1 = arith.constant 1 : index
  %c0_1 = arith.constant 0 : index
  %24 = arith.subi %23, %22 : index
  %25 = arith.addi %24, %c1 : index
  %26 = arith.divsi %25, %c1 : index
  %27 = arith.cmpi sgt, %26, %c0_1 : index
  %28 = arith.select %27, %26, %c0_1 : index
  %29 = fir.shape %28 : (index) -> !fir.shape<1>
  %30 = hlfir.designate %17 (%22:%23:%c1)  shape %29 : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
  %31 = fir.rebox %30 : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>>
  fir.store %31 to %11#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
  %32 = fir.load %1#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
  %33 = fir.load %5#0 : !fir.ref<i32>
  %34 = fir.convert %33 : (i32) -> i64
  %35 = fir.load %3#0 : !fir.ref<i32>
  %36 = fir.convert %35 : (i32) -> i64
  %37 = fir.convert %34 : (i64) -> index
  %38 = fir.convert %36 : (i64) -> index
  %c1_2 = arith.constant 1 : index
  %c0_3 = arith.constant 0 : index
  %39 = arith.subi %38, %37 : index
  %40 = arith.addi %39, %c1_2 : index
  %41 = arith.divsi %40, %c1_2 : index
  %42 = arith.cmpi sgt, %41, %c0_3 : index
  %43 = arith.select %42, %41, %c0_3 : index
  %44 = fir.shape %43 : (index) -> !fir.shape<1>
  %45 = hlfir.designate %32 (%37:%38:%c1_2)  shape %44 : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
  %46 = fir.rebox %45 : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>>
  fir.store %46 to %16#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
  %47 = fir.load %16#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
  %48 = fir.load %11#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
  hlfir.assign %47 to %48 : !fir.box<!fir.ptr<!fir.array<?xi32>>>, !fir.box<!fir.ptr<!fir.array<?xi32>>>
  return
}

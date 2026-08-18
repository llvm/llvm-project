//  Test inlining of hlfir.assign for array sections with the same runtime stride on LHS and RHS
// RUN: fir-opt --inline-hlfir-assign %s | FileCheck %s

// Fortran source:
//   subroutine testcase(a, n, step)
//     implicit none
//     integer, intent(in) :: n, step
//     real, intent(inout) :: a(n)
//     integer :: lo1, hi1, lo2, hi2
//     lo1 = 1
//     hi1 = max(n / 2, 1)
//     lo2 = 2
//     hi2 = n
//     a(lo1:hi1:step) = a(lo2:hi2:step)
//   end subroutine testcase
//

// CHECK-LABEL: func.func @_QPtestcase
// CHECK-SAME: %[[A:.*]]: !fir.ref<!fir.array<?xf32>>

// CHECK: %{{.*}} = hlfir.designate
// CHECK: %{{.*}} = hlfir.designate
// CHECK: %[[LHS_ADDR:.*]] = fir.box_addr %{{.*}}
// CHECK: %[[LHS_BASE:.*]] = fir.convert %[[LHS_ADDR]] : (!fir.ref<!fir.array<?xf32>>) -> i64
// CHECK: %[[LHS_ELESZ:.*]] = fir.box_elesize %{{.*}}
// CHECK: %[[LHS_DIMS:.*]]:3 = fir.box_dims %{{.*}}, %{{.*}}
// CHECK: %{{.*}} = arith.subi %[[LHS_DIMS]]#1, %{{.*}} : index
// CHECK: %{{.*}} = arith.muli %{{.*}}, %[[LHS_DIMS]]#2 : index
// CHECK: %{{.*}} = arith.cmpi slt, %[[LHS_DIMS]]#2, %{{.*}} : index
// CHECK: %{{.*}} = arith.select %{{.*}}, %{{.*}}, %{{.*}} : index
// CHECK: %{{.*}} = arith.select %{{.*}}, %{{.*}}, %{{.*}} : index
// CHECK: %{{.*}} = arith.subi %[[LHS_ELESZ]], %{{.*}} : index
// CHECK: %{{.*}} = arith.addi %{{.*}}, %{{.*}} : index
// CHECK: %{{.*}} = fir.convert %{{.*}} : (index) -> i64
// CHECK: %{{.*}} = fir.convert %{{.*}} : (index) -> i64
// CHECK: %[[LHS_START:.*]] = arith.addi %[[LHS_BASE]], %{{.*}} : i64
// CHECK: %[[LHS_END:.*]] = arith.addi %[[LHS_BASE]], %{{.*}} : i64
// CHECK: %[[RHS_ADDR:.*]] = fir.box_addr %{{.*}}
// CHECK: %[[RHS_BASE:.*]] = fir.convert %[[RHS_ADDR]] : (!fir.ref<!fir.array<?xf32>>) -> i64
// CHECK: %[[RHS_ELESZ:.*]] = fir.box_elesize %{{.*}}
// CHECK: %[[RHS_DIMS:.*]]:3 = fir.box_dims %{{.*}}, %{{.*}}
// CHECK: %{{.*}} = arith.subi %[[RHS_DIMS]]#1, %{{.*}} : index
// CHECK: %{{.*}} = arith.muli %{{.*}}, %[[RHS_DIMS]]#2 : index
// CHECK: %{{.*}} = arith.cmpi slt, %[[RHS_DIMS]]#2, %{{.*}} : index
// CHECK: %{{.*}} = arith.select %{{.*}}, %{{.*}}, %{{.*}} : index
// CHECK: %{{.*}} = arith.select %{{.*}}, %{{.*}}, %{{.*}} : index
// CHECK: %{{.*}} = arith.subi %[[RHS_ELESZ]], %{{.*}} : index
// CHECK: %{{.*}} = arith.addi %{{.*}}, %{{.*}} : index
// CHECK: %{{.*}} = fir.convert %{{.*}} : (index) -> i64
// CHECK: %{{.*}} = fir.convert %{{.*}} : (index) -> i64
// CHECK: %[[RHS_START:.*]] = arith.addi %[[RHS_BASE]], %{{.*}} : i64
// CHECK: %[[RHS_END:.*]] = arith.addi %[[RHS_BASE]], %{{.*}} : i64
// CHECK: %[[CMP1:.*]] = arith.cmpi ult, %[[LHS_END]], %[[RHS_START]] : i64
// CHECK: %[[CMP2:.*]] = arith.cmpi ult, %[[RHS_END]], %[[LHS_START]] : i64
// CHECK: %[[DISJOINT:.*]] = arith.ori %[[CMP1]], %[[CMP2]] : i1
// CHECK: fir.if %[[DISJOINT]] {
// CHECK:   fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} unordered {
// CHECK:     %{{.*}} = hlfir.designate %{{.*}} (%{{.*}})
// CHECK:     %{{.*}} = fir.load %{{.*}}
// CHECK:     %{{.*}} = hlfir.designate %{{.*}} (%{{.*}})
// CHECK:     hlfir.assign %{{.*}} to %{{.*}}
// CHECK:   }
// CHECK: } else {
// CHECK:   %[[ALLOC:.*]] = fir.allocmem !fir.array<?xf32>, %{{.*}} {bindc_name = ".tmp", uniq_name = ""}
// CHECK:   %[[TEMP:.*]]:2 = hlfir.declare %[[ALLOC]](%{{.*}}) {uniq_name = ".tmp"}
// CHECK:   fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} unordered {
// CHECK:     %{{.*}} = hlfir.designate %{{.*}} (%{{.*}})
// CHECK:     %{{.*}} = fir.load %{{.*}}
// CHECK:     %{{.*}} = hlfir.designate %[[TEMP]]#0 (%{{.*}})
// CHECK:     hlfir.assign %{{.*}} to %{{.*}} temporary_lhs
// CHECK:   }
// CHECK:   fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} unordered {
// CHECK:     %{{.*}} = hlfir.designate %[[TEMP]]#0 (%{{.*}})
// CHECK:     %{{.*}} = fir.load %{{.*}}
// CHECK:     %{{.*}} = hlfir.designate %{{.*}} (%{{.*}})
// CHECK:     hlfir.assign %{{.*}} to %{{.*}}
// CHECK:   }
// CHECK:   fir.freemem %{{.*}}
// CHECK: }

func.func @_QPtestcase(%arg0: !fir.ref<!fir.array<?xf32>> {fir.bindc_name = "a"}, %arg1: !fir.ref<i32> {fir.bindc_name = "n"}, %arg2: !fir.ref<i32> {fir.bindc_name = "step"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg1 dummy_scope %0 arg 2 {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFtestcaseEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %2 = fir.alloca i32 {bindc_name = "hi1", uniq_name = "_QFtestcaseEhi1"}
  %3:2 = hlfir.declare %2 {uniq_name = "_QFtestcaseEhi1"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %4 = fir.alloca i32 {bindc_name = "hi2", uniq_name = "_QFtestcaseEhi2"}
  %5:2 = hlfir.declare %4 {uniq_name = "_QFtestcaseEhi2"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %6 = fir.alloca i32 {bindc_name = "lo1", uniq_name = "_QFtestcaseElo1"}
  %7:2 = hlfir.declare %6 {uniq_name = "_QFtestcaseElo1"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %8 = fir.alloca i32 {bindc_name = "lo2", uniq_name = "_QFtestcaseElo2"}
  %9:2 = hlfir.declare %8 {uniq_name = "_QFtestcaseElo2"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %10:2 = hlfir.declare %arg2 dummy_scope %0 arg 3 {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFtestcaseEstep"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %11 = fir.load %1#0 : !fir.ref<i32>
  %12 = fir.convert %11 : (i32) -> i64
  %13 = fir.convert %12 : (i64) -> index
  %c0 = arith.constant 0 : index
  %14 = arith.cmpi sgt, %13, %c0 : index
  %15 = arith.select %14, %13, %c0 : index
  %16 = fir.shape %15 : (index) -> !fir.shape<1>
  %17:2 = hlfir.declare %arg0(%16) dummy_scope %0 arg 1 {fortran_attrs = #fir.var_attrs<intent_inout>, uniq_name = "_QFtestcaseEa"} : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>)
  %c1_i32 = arith.constant 1 : i32
  hlfir.assign %c1_i32 to %7#0 : i32, !fir.ref<i32>
  %18 = fir.load %1#0 : !fir.ref<i32>
  %c2_i32 = arith.constant 2 : i32
  %19 = arith.divsi %18, %c2_i32 : i32
  %c1_i32_0 = arith.constant 1 : i32
  %20 = arith.maxsi %19, %c1_i32_0 : i32
  hlfir.assign %20 to %3#0 : i32, !fir.ref<i32>
  %c2_i32_1 = arith.constant 2 : i32
  hlfir.assign %c2_i32_1 to %9#0 : i32, !fir.ref<i32>
  %21 = fir.load %1#0 : !fir.ref<i32>
  hlfir.assign %21 to %5#0 : i32, !fir.ref<i32>
  %22 = fir.load %9#0 : !fir.ref<i32>
  %23 = fir.convert %22 : (i32) -> i64
  %24 = fir.load %5#0 : !fir.ref<i32>
  %25 = fir.convert %24 : (i32) -> i64
  %26 = fir.convert %23 : (i64) -> index
  %27 = fir.convert %25 : (i64) -> index
  %28 = fir.load %10#0 : !fir.ref<i32>
  %29 = fir.convert %28 : (i32) -> i64
  %30 = fir.convert %29 : (i64) -> index
  %c0_2 = arith.constant 0 : index
  %31 = arith.subi %27, %26 : index
  %32 = arith.addi %31, %30 : index
  %33 = arith.divsi %32, %30 : index
  %34 = arith.cmpi sgt, %33, %c0_2 : index
  %35 = arith.select %34, %33, %c0_2 : index
  %36 = fir.shape %35 : (index) -> !fir.shape<1>
  %37 = hlfir.designate %17#0 (%26:%27:%30)  shape %36 : (!fir.box<!fir.array<?xf32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
  %38 = fir.load %7#0 : !fir.ref<i32>
  %39 = fir.convert %38 : (i32) -> i64
  %40 = fir.load %3#0 : !fir.ref<i32>
  %41 = fir.convert %40 : (i32) -> i64
  %42 = fir.convert %39 : (i64) -> index
  %43 = fir.convert %41 : (i64) -> index
  %44 = fir.load %10#0 : !fir.ref<i32>
  %45 = fir.convert %44 : (i32) -> i64
  %46 = fir.convert %45 : (i64) -> index
  %c0_3 = arith.constant 0 : index
  %47 = arith.subi %43, %42 : index
  %48 = arith.addi %47, %46 : index
  %49 = arith.divsi %48, %46 : index
  %50 = arith.cmpi sgt, %49, %c0_3 : index
  %51 = arith.select %50, %49, %c0_3 : index
  %52 = fir.shape %51 : (index) -> !fir.shape<1>
  %53 = hlfir.designate %17#0 (%42:%43:%46)  shape %52 : (!fir.box<!fir.array<?xf32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
  hlfir.assign %37 to %53 : !fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>
  return
}

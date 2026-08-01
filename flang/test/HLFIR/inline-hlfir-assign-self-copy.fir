// Test inlining of hlfir.assign for self-copy array section

// RUN: fir-opt --inline-hlfir-assign %s | FileCheck %s

// Fortran source code:
// subroutine self_copy_ew(field, xm, ym, levs, xh, yh)
//    integer, intent(in)    :: xm, ym, levs, xh, yh
//    real,    intent(inout) :: field(xm + 2 * xh, ym + 2 * yh, levs)
//    integer :: x1, x2, x5, x6 
//    x1 = 1
//    x2 = xh
//    x5 = xm + 1
//    x6 = xm + xh
//    field(x1:x2, :, :) = field(x5:x6, :, :) 
//  end subroutine self_copy_ew

// CHECK-LABEL: func.func @_QPself_copy_ew
// CHECK-SAME: %[[ARGF:.*]]: !fir.ref<!fir.array<?x?x?xf32>> {fir.bindc_name = "field"}

// CHECK: hlfir.declare %[[ARGF]](%{{.*}}) {{.*}} uniq_name = "_QFself_copy_ewEfield"

// Index-based runtime disjointness check:
// CHECK: %[[RHS:.*]] = hlfir.designate %{{.*}}#0 (%{{.*}}:%{{.*}}:%{{.*}}, %{{.*}}:%{{.*}}:%{{.*}}, %{{.*}}:%{{.*}}:%{{.*}}) shape %{{.*}}
// CHECK: %[[LHS:.*]] = hlfir.designate %{{.*}}#0 (%{{.*}}:%{{.*}}:%{{.*}}, %{{.*}}:%{{.*}}:%{{.*}}, %{{.*}}:%{{.*}}:%{{.*}}) shape %{{.*}}
// CHECK: %[[CMP1:.*]] = arith.cmpi slt, %{{.*}}, %{{.*}} : index
// CHECK: %[[CMP2:.*]] = arith.cmpi slt, %{{.*}}, %{{.*}} : index
// CHECK: %[[DIM1_DISJ:.*]] = arith.ori %[[CMP1]], %[[CMP2]] : i1
// CHECK: %[[CMP3:.*]] = arith.cmpi slt, %{{.*}}, %{{.*}} : index
// CHECK: %[[CMP4:.*]] = arith.cmpi slt, %{{.*}}, %{{.*}} : index
// CHECK: %[[DIM2_DISJ:.*]] = arith.ori %[[CMP3]], %[[CMP4]] : i1
// CHECK: %[[DISJ12:.*]] = arith.ori %[[DIM1_DISJ]], %[[DIM2_DISJ]] : i1
// CHECK: %[[CMP5:.*]] = arith.cmpi slt, %{{.*}}, %{{.*}} : index
// CHECK: %[[CMP6:.*]] = arith.cmpi slt, %{{.*}}, %{{.*}} : index
// CHECK: %[[DIM3_DISJ:.*]] = arith.ori %[[CMP5]], %[[CMP6]] : i1
// CHECK: %[[DISJOINT:.*]] = arith.ori %[[DISJ12]], %[[DIM3_DISJ]] : i1
// CHECK: fir.if %[[DISJOINT]] {
// CHECK:   fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} unordered {
// CHECK:     fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} unordered {
// CHECK:       fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} unordered {
// CHECK:         %{{.*}} = hlfir.designate %[[RHS]] (%{{.*}}, %{{.*}}, %{{.*}})
// CHECK:         %{{.*}} = fir.load %{{.*}}
// CHECK:         %{{.*}} = hlfir.designate %[[LHS]] (%{{.*}}, %{{.*}}, %{{.*}})
// CHECK:         hlfir.assign %{{.*}} to %{{.*}} : f32, !fir.ref<f32>
// CHECK:       }
// CHECK:     }
// CHECK:   }
// CHECK: } else {
// CHECK:   %[[ALLOC:.*]] = fir.allocmem !fir.array<?x?x?xf32>{{.*}}
// CHECK:   %[[TEMP:.*]]:2 = hlfir.declare %[[ALLOC]](%{{.*}}) {uniq_name = ".tmp"}
// CHECK:   fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} unordered {
// CHECK:     fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} unordered {
// CHECK:       fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} unordered {
// CHECK:         %{{.*}} = hlfir.designate %[[RHS]] (%{{.*}}, %{{.*}}, %{{.*}})
// CHECK:         %{{.*}} = fir.load %{{.*}}
// CHECK:         %{{.*}} = hlfir.designate %[[TEMP]]#0 (%{{.*}}, %{{.*}}, %{{.*}})
// CHECK:         hlfir.assign %{{.*}} to %{{.*}} temporary_lhs : f32, !fir.ref<f32>
// CHECK:       }
// CHECK:     }
// CHECK:   }
// CHECK:   fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} unordered {
// CHECK:     fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} unordered {
// CHECK:       fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} unordered {
// CHECK:         %{{.*}} = hlfir.designate %[[TEMP]]#0 (%{{.*}}, %{{.*}}, %{{.*}})
// CHECK:         %{{.*}} = fir.load %{{.*}}
// CHECK:         %{{.*}} = hlfir.designate %[[LHS]] (%{{.*}}, %{{.*}}, %{{.*}})
// CHECK:         hlfir.assign %{{.*}} to %{{.*}} : f32, !fir.ref<f32>
// CHECK:       }
// CHECK:     }
// CHECK:   }
// CHECK:   fir.freemem %{{.*}}
// CHECK: }

func.func @_QPself_copy_ew(%arg0: !fir.ref<!fir.array<?x?x?xf32>> {fir.bindc_name = "field"}, %arg1: !fir.ref<i32> {fir.bindc_name = "xm"}, %arg2: !fir.ref<i32> {fir.bindc_name = "ym"}, %arg3: !fir.ref<i32> {fir.bindc_name = "levs"}, %arg4: !fir.ref<i32> {fir.bindc_name = "xh"}, %arg5: !fir.ref<i32> {fir.bindc_name = "yh"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg1 dummy_scope %0 arg 2 {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFself_copy_ewExm"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %2:2 = hlfir.declare %arg4 dummy_scope %0 arg 5 {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFself_copy_ewExh"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %3:2 = hlfir.declare %arg2 dummy_scope %0 arg 3 {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFself_copy_ewEym"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %4:2 = hlfir.declare %arg5 dummy_scope %0 arg 6 {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFself_copy_ewEyh"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %5:2 = hlfir.declare %arg3 dummy_scope %0 arg 4 {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFself_copy_ewElevs"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %6 = fir.alloca i32 {bindc_name = "x1", uniq_name = "_QFself_copy_ewEx1"}
  %7:2 = hlfir.declare %6 {uniq_name = "_QFself_copy_ewEx1"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %8 = fir.alloca i32 {bindc_name = "x2", uniq_name = "_QFself_copy_ewEx2"}
  %9:2 = hlfir.declare %8 {uniq_name = "_QFself_copy_ewEx2"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %10 = fir.alloca i32 {bindc_name = "x5", uniq_name = "_QFself_copy_ewEx5"}
  %11:2 = hlfir.declare %10 {uniq_name = "_QFself_copy_ewEx5"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %12 = fir.alloca i32 {bindc_name = "x6", uniq_name = "_QFself_copy_ewEx6"}
  %13:2 = hlfir.declare %12 {uniq_name = "_QFself_copy_ewEx6"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %14 = fir.load %1#0 : !fir.ref<i32>
  %c2_i32 = arith.constant 2 : i32
  %15 = fir.load %2#0 : !fir.ref<i32>
  %16 = arith.muli %c2_i32, %15 : i32
  %17 = arith.addi %14, %16 : i32
  %18 = fir.convert %17 : (i32) -> i64
  %19 = fir.convert %18 : (i64) -> index
  %c0 = arith.constant 0 : index
  %20 = arith.cmpi sgt, %19, %c0 : index
  %21 = arith.select %20, %19, %c0 : index
  %22 = fir.load %3#0 : !fir.ref<i32>
  %c2_i32_0 = arith.constant 2 : i32
  %23 = fir.load %4#0 : !fir.ref<i32>
  %24 = arith.muli %c2_i32_0, %23 : i32
  %25 = arith.addi %22, %24 : i32
  %26 = fir.convert %25 : (i32) -> i64
  %27 = fir.convert %26 : (i64) -> index
  %c0_1 = arith.constant 0 : index
  %28 = arith.cmpi sgt, %27, %c0_1 : index
  %29 = arith.select %28, %27, %c0_1 : index
  %30 = fir.load %5#0 : !fir.ref<i32>
  %31 = fir.convert %30 : (i32) -> i64
  %32 = fir.convert %31 : (i64) -> index
  %c0_2 = arith.constant 0 : index
  %33 = arith.cmpi sgt, %32, %c0_2 : index
  %34 = arith.select %33, %32, %c0_2 : index
  %35 = fir.shape %21, %29, %34 : (index, index, index) -> !fir.shape<3>
  %36:2 = hlfir.declare %arg0(%35) dummy_scope %0 arg 1 {fortran_attrs = #fir.var_attrs<intent_inout>, uniq_name = "_QFself_copy_ewEfield"} : (!fir.ref<!fir.array<?x?x?xf32>>, !fir.shape<3>, !fir.dscope) -> (!fir.box<!fir.array<?x?x?xf32>>, !fir.ref<!fir.array<?x?x?xf32>>)
  %c1_i32 = arith.constant 1 : i32
  hlfir.assign %c1_i32 to %7#0 : i32, !fir.ref<i32>
  %37 = fir.load %2#0 : !fir.ref<i32>
  hlfir.assign %37 to %9#0 : i32, !fir.ref<i32>
  %38 = fir.load %1#0 : !fir.ref<i32>
  %c1_i32_3 = arith.constant 1 : i32
  %39 = arith.addi %38, %c1_i32_3 : i32
  hlfir.assign %39 to %11#0 : i32, !fir.ref<i32>
  %40 = fir.load %1#0 : !fir.ref<i32>
  %41 = fir.load %2#0 : !fir.ref<i32>
  %42 = arith.addi %40, %41 : i32
  hlfir.assign %42 to %13#0 : i32, !fir.ref<i32>
  %43 = fir.load %11#0 : !fir.ref<i32>
  %44 = fir.convert %43 : (i32) -> i64
  %45 = fir.load %13#0 : !fir.ref<i32>
  %46 = fir.convert %45 : (i32) -> i64
  %47 = fir.convert %44 : (i64) -> index
  %48 = fir.convert %46 : (i64) -> index
  %c1 = arith.constant 1 : index
  %c0_4 = arith.constant 0 : index
  %49 = arith.subi %48, %47 : index
  %50 = arith.addi %49, %c1 : index
  %51 = arith.divsi %50, %c1 : index
  %52 = arith.cmpi sgt, %51, %c0_4 : index
  %53 = arith.select %52, %51, %c0_4 : index
  %c1_5 = arith.constant 1 : index
  %c1_6 = arith.constant 1 : index
  %c0_7 = arith.constant 0 : index
  %54 = arith.subi %29, %c1_5 : index
  %55 = arith.addi %54, %c1_6 : index
  %56 = arith.divsi %55, %c1_6 : index
  %57 = arith.cmpi sgt, %56, %c0_7 : index
  %58 = arith.select %57, %56, %c0_7 : index
  %c1_8 = arith.constant 1 : index
  %c0_9 = arith.constant 0 : index
  %59 = arith.subi %34, %c1_5 : index
  %60 = arith.addi %59, %c1_8 : index
  %61 = arith.divsi %60, %c1_8 : index
  %62 = arith.cmpi sgt, %61, %c0_9 : index
  %63 = arith.select %62, %61, %c0_9 : index
  %64 = fir.shape %53, %58, %63 : (index, index, index) -> !fir.shape<3>
  %65 = hlfir.designate %36#0 (%47:%48:%c1, %c1_5:%29:%c1_6, %c1_5:%34:%c1_8)  shape %64 : (!fir.box<!fir.array<?x?x?xf32>>, index, index, index, index, index, index, index, index, index, !fir.shape<3>) -> !fir.box<!fir.array<?x?x?xf32>>
  %66 = fir.load %7#0 : !fir.ref<i32>
  %67 = fir.convert %66 : (i32) -> i64
  %68 = fir.load %9#0 : !fir.ref<i32>
  %69 = fir.convert %68 : (i32) -> i64
  %70 = fir.convert %67 : (i64) -> index
  %71 = fir.convert %69 : (i64) -> index
  %c1_10 = arith.constant 1 : index
  %c0_11 = arith.constant 0 : index
  %72 = arith.subi %71, %70 : index
  %73 = arith.addi %72, %c1_10 : index
  %74 = arith.divsi %73, %c1_10 : index
  %75 = arith.cmpi sgt, %74, %c0_11 : index
  %76 = arith.select %75, %74, %c0_11 : index
  %c1_12 = arith.constant 1 : index
  %c1_13 = arith.constant 1 : index
  %c0_14 = arith.constant 0 : index
  %77 = arith.subi %29, %c1_12 : index
  %78 = arith.addi %77, %c1_13 : index
  %79 = arith.divsi %78, %c1_13 : index
  %80 = arith.cmpi sgt, %79, %c0_14 : index
  %81 = arith.select %80, %79, %c0_14 : index
  %c1_15 = arith.constant 1 : index
  %c0_16 = arith.constant 0 : index
  %82 = arith.subi %34, %c1_12 : index
  %83 = arith.addi %82, %c1_15 : index
  %84 = arith.divsi %83, %c1_15 : index
  %85 = arith.cmpi sgt, %84, %c0_16 : index
  %86 = arith.select %85, %84, %c0_16 : index
  %87 = fir.shape %76, %81, %86 : (index, index, index) -> !fir.shape<3>
  %88 = hlfir.designate %36#0 (%70:%71:%c1_10, %c1_12:%29:%c1_13, %c1_12:%34:%c1_15)  shape %87 : (!fir.box<!fir.array<?x?x?xf32>>, index, index, index, index, index, index, index, index, index, !fir.shape<3>) -> !fir.box<!fir.array<?x?x?xf32>>
  hlfir.assign %65 to %88 : !fir.box<!fir.array<?x?x?xf32>>, !fir.box<!fir.array<?x?x?xf32>>
  return
}

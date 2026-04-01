// RUN: fir-opt --opt-bufferization %s | FileCheck %s

// SUBROUTINE TEST(N,M,K,U,V)
// IMPLICIT NONE
// INTEGER(KIND=4) :: N,M,K
// REAL(KIND=4), DIMENSION(:,:),   POINTER, CONTIGUOUS,INTENT(IN) :: U
// REAL(KIND=4), DIMENSION(:,:),   POINTER, CONTIGUOUS,INTENT(OUT) :: V
// INTEGER (KIND=4)  :: J
//
// !dir$ ivdep
// DO J=1,K
//   V(N:M,J) = 2.0*U(N:M,J)
// ENDDO
// END SUBROUTINE TEST

func.func @_QPtest(%arg0: !fir.ref<i32> {fir.bindc_name = "n"}, %arg1: !fir.ref<i32> {fir.bindc_name = "m"}, %arg2: !fir.ref<i32> {fir.bindc_name = "k"}, %arg3: !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>> {fir.bindc_name = "u", fir.contiguous}, %arg4: !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>> {fir.bindc_name = "v", fir.contiguous}) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 2.000000e+00 : f32
  %c1 = arith.constant 1 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.alloca i32 {bindc_name = "j", uniq_name = "_QFtestEj"}
  %2:2 = hlfir.declare %1 {uniq_name = "_QFtestEj"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %3:2 = hlfir.declare %arg2 dummy_scope %0 arg 3 {uniq_name = "_QFtestEk"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %4:2 = hlfir.declare %arg1 dummy_scope %0 arg 2 {uniq_name = "_QFtestEm"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %5:2 = hlfir.declare %arg0 dummy_scope %0 arg 1 {uniq_name = "_QFtestEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %6:2 = hlfir.declare %arg3 dummy_scope %0 arg 4 {fortran_attrs = #fir.var_attrs<contiguous, intent_in, pointer>, uniq_name = "_QFtestEu"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>)
  %7:2 = hlfir.declare %arg4 dummy_scope %0 arg 5 {fortran_attrs = #fir.var_attrs<contiguous, intent_out, pointer>, uniq_name = "_QFtestEv"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>)
  %8 = fir.load %3#0 : !fir.ref<i32>
  %9 = fir.convert %8 : (i32) -> index
  %10 = fir.convert %c1 : (index) -> i32
  %11 = fir.do_loop %arg5 = %c1 to %9 step %c1 iter_args(%arg6 = %10) -> (i32) attributes {loopAnnotation = #llvm.loop_annotation<parallelAccesses = <id = distinct[0]<>>>} {
    fir.store %arg6 to %2#0 {accessGroups = [#llvm.access_group<id = distinct[0]<>>]} : !fir.ref<i32>
    %12 = fir.load %6#0 {accessGroups = [#llvm.access_group<id = distinct[0]<>>]} : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>
    %13 = fir.load %5#0 {accessGroups = [#llvm.access_group<id = distinct[0]<>>]} : !fir.ref<i32>
    %14 = fir.load %4#0 {accessGroups = [#llvm.access_group<id = distinct[0]<>>]} : !fir.ref<i32>
    %15 = fir.convert %13 : (i32) -> index
    %16 = fir.convert %14 : (i32) -> index
    %17 = arith.subi %16, %15 : index
    %18 = arith.addi %17, %c1 : index
    %19 = arith.cmpi sgt, %18, %c0 : index
    %20 = arith.select %19, %18, %c0 : index
    %21 = fir.load %2#0 {accessGroups = [#llvm.access_group<id = distinct[0]<>>]} : !fir.ref<i32>
    %22 = fir.convert %21 : (i32) -> i64
    %23 = fir.shape %20 : (index) -> !fir.shape<1>
    %24 = hlfir.designate %12 (%15:%16:%c1, %22)  shape %23 : (!fir.box<!fir.ptr<!fir.array<?x?xf32>>>, index, index, index, i64, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
    %25 = hlfir.elemental %23 unordered : (!fir.shape<1>) -> !hlfir.expr<?xf32> {
    ^bb0(%arg7: index):
      %30 = hlfir.designate %24 (%arg7)  : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
      %31 = fir.load %30 {accessGroups = [#llvm.access_group<id = distinct[0]<>>]} : !fir.ref<f32>
      %32 = arith.mulf %31, %cst fastmath<contract> : f32
      hlfir.yield_element %32 : f32
    }
    %26 = fir.load %7#0 {accessGroups = [#llvm.access_group<id = distinct[0]<>>]} : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>
    %27 = hlfir.designate %26 (%15:%16:%c1, %22)  shape %23 : (!fir.box<!fir.ptr<!fir.array<?x?xf32>>>, index, index, index, i64, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
    hlfir.assign %25 to %27 {access_groups = [#llvm.access_group<id = distinct[0]<>>]} : !hlfir.expr<?xf32>, !fir.box<!fir.array<?xf32>>
    hlfir.destroy %25 : !hlfir.expr<?xf32>
    %28 = fir.load %2#0 {accessGroups = [#llvm.access_group<id = distinct[0]<>>]} : !fir.ref<i32>
    %29 = arith.addi %28, %10 overflow<nsw> : i32
    fir.result %29 : i32
  }
  fir.store %11 to %2#0 : !fir.ref<i32>
  return
}
// CHECK: #[[$ATTR_0:.+]] = #llvm.access_group<id = distinct[0]<>>
// CHECK: #[[$ATTR_1:.+]] = #llvm.loop_annotation<parallelAccesses = #[[$ATTR_0]]>
// CHECK-LABEL:   func.func @_QPtest(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<i32> {fir.bindc_name = "n"},
// CHECK-SAME:      %[[ARG1:.*]]: !fir.ref<i32> {fir.bindc_name = "m"},
// CHECK-SAME:      %[[ARG2:.*]]: !fir.ref<i32> {fir.bindc_name = "k"},
// CHECK-SAME:      %[[ARG3:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>> {fir.bindc_name = "u", fir.contiguous},
// CHECK-SAME:      %[[ARG4:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>> {fir.bindc_name = "v", fir.contiguous}) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 2.000000e+00 : f32
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 1 : index
// CHECK:           %[[DUMMY_SCOPE_0:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[ALLOCA_0:.*]] = fir.alloca i32 {bindc_name = "j", uniq_name = "_QFtestEj"}
// CHECK:           %[[DECLARE_0:.*]]:2 = hlfir.declare %[[ALLOCA_0]] {uniq_name = "_QFtestEj"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK:           %[[DECLARE_1:.*]]:2 = hlfir.declare %[[ARG2]] dummy_scope %[[DUMMY_SCOPE_0]] arg 3 {uniq_name = "_QFtestEk"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK:           %[[DECLARE_2:.*]]:2 = hlfir.declare %[[ARG1]] dummy_scope %[[DUMMY_SCOPE_0]] arg 2 {uniq_name = "_QFtestEm"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK:           %[[DECLARE_3:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[DUMMY_SCOPE_0]] arg 1 {uniq_name = "_QFtestEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK:           %[[DECLARE_4:.*]]:2 = hlfir.declare %[[ARG3]] dummy_scope %[[DUMMY_SCOPE_0]] arg 4 {fortran_attrs = #fir.var_attrs<contiguous, intent_in, pointer>, uniq_name = "_QFtestEu"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>)
// CHECK:           %[[DECLARE_5:.*]]:2 = hlfir.declare %[[ARG4]] dummy_scope %[[DUMMY_SCOPE_0]] arg 5 {fortran_attrs = #fir.var_attrs<contiguous, intent_out, pointer>, uniq_name = "_QFtestEv"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>)
// CHECK:           %[[LOAD_0:.*]] = fir.load %[[DECLARE_1]]#0 : !fir.ref<i32>
// CHECK:           %[[CONVERT_0:.*]] = fir.convert %[[LOAD_0]] : (i32) -> index
// CHECK:           %[[CONVERT_1:.*]] = fir.convert %[[CONSTANT_2]] : (index) -> i32
// CHECK:           %[[DO_LOOP_0:.*]] = fir.do_loop %[[VAL_0:.*]] = %[[CONSTANT_2]] to %[[CONVERT_0]] step %[[CONSTANT_2]] iter_args(%[[VAL_1:.*]] = %[[CONVERT_1]]) -> (i32) attributes {loopAnnotation = #[[$ATTR_1]]} {
// CHECK:             fir.store %[[VAL_1]] to %[[DECLARE_0]]#0 {accessGroups = [#[[$ATTR_0]]]} : !fir.ref<i32>
// CHECK:             %[[LOAD_1:.*]] = fir.load %[[DECLARE_4]]#0 {accessGroups = [#[[$ATTR_0]]]} : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>
// CHECK:             %[[LOAD_2:.*]] = fir.load %[[DECLARE_3]]#0 {accessGroups = [#[[$ATTR_0]]]} : !fir.ref<i32>
// CHECK:             %[[LOAD_3:.*]] = fir.load %[[DECLARE_2]]#0 {accessGroups = [#[[$ATTR_0]]]} : !fir.ref<i32>
// CHECK:             %[[CONVERT_2:.*]] = fir.convert %[[LOAD_2]] : (i32) -> index
// CHECK:             %[[CONVERT_3:.*]] = fir.convert %[[LOAD_3]] : (i32) -> index
// CHECK:             %[[SUBI_0:.*]] = arith.subi %[[CONVERT_3]], %[[CONVERT_2]] : index
// CHECK:             %[[ADDI_0:.*]] = arith.addi %[[SUBI_0]], %[[CONSTANT_2]] : index
// CHECK:             %[[CMPI_0:.*]] = arith.cmpi sgt, %[[ADDI_0]], %[[CONSTANT_0]] : index
// CHECK:             %[[SELECT_0:.*]] = arith.select %[[CMPI_0]], %[[ADDI_0]], %[[CONSTANT_0]] : index
// CHECK:             %[[LOAD_4:.*]] = fir.load %[[DECLARE_0]]#0 {accessGroups = [#[[$ATTR_0]]]} : !fir.ref<i32>
// CHECK:             %[[CONVERT_4:.*]] = fir.convert %[[LOAD_4]] : (i32) -> i64
// CHECK:             %[[SHAPE_0:.*]] = fir.shape %[[SELECT_0]] : (index) -> !fir.shape<1>
// CHECK:             %[[DESIGNATE_0:.*]] = hlfir.designate %[[LOAD_1]] (%[[CONVERT_2]]:%[[CONVERT_3]]:%[[CONSTANT_2]], %[[CONVERT_4]])  shape %[[SHAPE_0]] : (!fir.box<!fir.ptr<!fir.array<?x?xf32>>>, index, index, index, i64, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
// CHECK:             %[[ELEMENTAL_0:.*]] = hlfir.elemental %[[SHAPE_0]] unordered : (!fir.shape<1>) -> !hlfir.expr<?xf32> {
// CHECK:             ^bb0(%[[VAL_2:.*]]: index):
// CHECK:               %[[DESIGNATE_1:.*]] = hlfir.designate %[[DESIGNATE_0]] (%[[VAL_2]])  : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
// CHECK:               %[[LOAD_5:.*]] = fir.load %[[DESIGNATE_1]] {accessGroups = [#[[$ATTR_0]]]} : !fir.ref<f32>
// CHECK:               %[[MULF_0:.*]] = arith.mulf %[[LOAD_5]], %[[CONSTANT_1]] fastmath<contract> : f32
// CHECK:               hlfir.yield_element %[[MULF_0]] : f32
// CHECK:             }
// CHECK:             %[[LOAD_6:.*]] = fir.load %[[DECLARE_5]]#0 {accessGroups = [#[[$ATTR_0]]]} : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>
// CHECK:             %[[DESIGNATE_2:.*]] = hlfir.designate %[[LOAD_6]] (%[[CONVERT_2]]:%[[CONVERT_3]]:%[[CONSTANT_2]], %[[CONVERT_4]])  shape %[[SHAPE_0]] : (!fir.box<!fir.ptr<!fir.array<?x?xf32>>>, index, index, index, i64, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
// CHECK:             hlfir.assign %[[ELEMENTAL_0]] to %[[DESIGNATE_2]] {access_groups = [#[[$ATTR_0]]]} : !hlfir.expr<?xf32>, !fir.box<!fir.array<?xf32>>
// CHECK:             hlfir.destroy %[[ELEMENTAL_0]] : !hlfir.expr<?xf32>
// CHECK:             %[[LOAD_7:.*]] = fir.load %[[DECLARE_0]]#0 {accessGroups = [#[[$ATTR_0]]]} : !fir.ref<i32>
// CHECK:             %[[ADDI_1:.*]] = arith.addi %[[LOAD_7]], %[[CONVERT_1]] overflow<nsw> : i32
// CHECK:             fir.result %[[ADDI_1]] : i32
// CHECK:           }
// CHECK:           fir.store %[[DO_LOOP_0]] to %[[DECLARE_0]]#0 : !fir.ref<i32>
// CHECK:           return
// CHECK:         }

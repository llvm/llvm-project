! RUN: %flang_fc1 -fopenmp -emit-hlfir %s -o - | FileCheck %s

subroutine substring_0(c)
  character(:), pointer :: c
  !$omp task depend(out:c(:))
  !$omp end task
end
! CHECK-LABEL:   func.func @_QPsubstring_0(
! CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>> {fir.bindc_name = "c"}) {
! CHECK:           %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[VAL_0]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFsubstring_0Ec"} : (!fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>)
! CHECK:           %[[VAL_2:.*]] = fir.load %[[VAL_1]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>
! CHECK:           %[[VAL_3:.*]] = fir.box_addr %[[VAL_2]] : (!fir.box<!fir.ptr<!fir.char<1,?>>>) -> !fir.ptr<!fir.char<1,?>>
! CHECK:           %[[VAL_4:.*]] = fir.load %[[VAL_1]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>
! CHECK:           %[[VAL_5:.*]] = fir.box_elesize %[[VAL_4]] : (!fir.box<!fir.ptr<!fir.char<1,?>>>) -> index
! CHECK:           %[[VAL_6:.*]] = fir.emboxchar %[[VAL_3]], %[[VAL_5]] : (!fir.ptr<!fir.char<1,?>>, index) -> !fir.boxchar<1>
! CHECK:           %[[VAL_7:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_8:.*]] = fir.load %[[VAL_1]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>
! CHECK:           %[[VAL_9:.*]] = fir.box_elesize %[[VAL_8]] : (!fir.box<!fir.ptr<!fir.char<1,?>>>) -> index
! CHECK:           %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (index) -> i64
! CHECK:           %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (i64) -> index
! CHECK:           %[[VAL_12:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_13:.*]] = arith.subi %[[VAL_11]], %[[VAL_7]] : index
! CHECK:           %[[VAL_14:.*]] = arith.addi %[[VAL_13]], %[[VAL_12]] : index
! CHECK:           %[[VAL_15:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_16:.*]] = arith.cmpi sgt, %[[VAL_14]], %[[VAL_15]] : index
! CHECK:           %[[VAL_17:.*]] = arith.select %[[VAL_16]], %[[VAL_14]], %[[VAL_15]] : index
! CHECK:           %[[VAL_18:.*]] = hlfir.designate %[[VAL_6]]  substr %[[VAL_7]], %[[VAL_11]]  typeparams %[[VAL_17]] : (!fir.boxchar<1>, index, index, index) -> !fir.boxchar<1>
! CHECK:           %[[VAL_19:.*]] = fir.box_addr %[[VAL_18]] : (!fir.boxchar<1>) -> !fir.ref<!fir.char<1,?>>
! CHECK:           omp.task depend(taskdependout -> %[[VAL_19]] : !fir.ref<!fir.char<1,?>>) {
! CHECK:             omp.terminator
! CHECK:           }
! CHECK:           return
! CHECK:         }

subroutine substring_1(c)
  character(:), pointer :: c
  !$omp task depend(out:c(2:))
  !$omp end task
end
! CHECK-LABEL:   func.func @_QPsubstring_1(
! CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>> {fir.bindc_name = "c"}) {
! CHECK:           %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[VAL_0]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFsubstring_1Ec"} : (!fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>)
! CHECK:           %[[VAL_2:.*]] = fir.load %[[VAL_1]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>
! CHECK:           %[[VAL_3:.*]] = fir.box_addr %[[VAL_2]] : (!fir.box<!fir.ptr<!fir.char<1,?>>>) -> !fir.ptr<!fir.char<1,?>>
! CHECK:           %[[VAL_4:.*]] = fir.load %[[VAL_1]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>
! CHECK:           %[[VAL_5:.*]] = fir.box_elesize %[[VAL_4]] : (!fir.box<!fir.ptr<!fir.char<1,?>>>) -> index
! CHECK:           %[[VAL_6:.*]] = fir.emboxchar %[[VAL_3]], %[[VAL_5]] : (!fir.ptr<!fir.char<1,?>>, index) -> !fir.boxchar<1>
! CHECK:           %[[VAL_7:.*]] = arith.constant 2 : index
! CHECK:           %[[VAL_8:.*]] = fir.load %[[VAL_1]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>
! CHECK:           %[[VAL_9:.*]] = fir.box_elesize %[[VAL_8]] : (!fir.box<!fir.ptr<!fir.char<1,?>>>) -> index
! CHECK:           %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (index) -> i64
! CHECK:           %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (i64) -> index
! CHECK:           %[[VAL_12:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_13:.*]] = arith.subi %[[VAL_11]], %[[VAL_7]] : index
! CHECK:           %[[VAL_14:.*]] = arith.addi %[[VAL_13]], %[[VAL_12]] : index
! CHECK:           %[[VAL_15:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_16:.*]] = arith.cmpi sgt, %[[VAL_14]], %[[VAL_15]] : index
! CHECK:           %[[VAL_17:.*]] = arith.select %[[VAL_16]], %[[VAL_14]], %[[VAL_15]] : index
! CHECK:           %[[VAL_18:.*]] = hlfir.designate %[[VAL_6]]  substr %[[VAL_7]], %[[VAL_11]]  typeparams %[[VAL_17]] : (!fir.boxchar<1>, index, index, index) -> !fir.boxchar<1>
! CHECK:           %[[VAL_19:.*]] = fir.box_addr %[[VAL_18]] : (!fir.boxchar<1>) -> !fir.ref<!fir.char<1,?>>
! CHECK:           omp.task depend(taskdependout -> %[[VAL_19]] : !fir.ref<!fir.char<1,?>>) {
! CHECK:             omp.terminator
! CHECK:           }
! CHECK:           return
! CHECK:         }

subroutine substring_2(c)
  character(:), pointer :: c
  !$omp task depend(out:c(:2))
  !$omp end task
end
! CHECK-LABEL:   func.func @_QPsubstring_2(
! CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>> {fir.bindc_name = "c"}) {
! CHECK:           %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[VAL_0]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFsubstring_2Ec"} : (!fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>)
! CHECK:           %[[VAL_2:.*]] = fir.load %[[VAL_1]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>
! CHECK:           %[[VAL_3:.*]] = fir.box_addr %[[VAL_2]] : (!fir.box<!fir.ptr<!fir.char<1,?>>>) -> !fir.ptr<!fir.char<1,?>>
! CHECK:           %[[VAL_4:.*]] = fir.load %[[VAL_1]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>
! CHECK:           %[[VAL_5:.*]] = fir.box_elesize %[[VAL_4]] : (!fir.box<!fir.ptr<!fir.char<1,?>>>) -> index
! CHECK:           %[[VAL_6:.*]] = fir.emboxchar %[[VAL_3]], %[[VAL_5]] : (!fir.ptr<!fir.char<1,?>>, index) -> !fir.boxchar<1>
! CHECK:           %[[VAL_7:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_8:.*]] = arith.constant 2 : index
! CHECK:           %[[VAL_9:.*]] = arith.constant 2 : index
! CHECK:           %[[VAL_10:.*]] = hlfir.designate %[[VAL_6]]  substr %[[VAL_7]], %[[VAL_8]]  typeparams %[[VAL_9]] : (!fir.boxchar<1>, index, index, index) -> !fir.ref<!fir.char<1,2>>
! CHECK:           omp.task depend(taskdependout -> %[[VAL_10]] : !fir.ref<!fir.char<1,2>>) {
! CHECK:             omp.terminator
! CHECK:           }
! CHECK:           return
! CHECK:         }

subroutine substring_4(c)
  character(:), pointer :: c
  !$omp task depend(out:c)
  !$omp end task
end
! CHECK-LABEL:   func.func @_QPsubstring_4(
! CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>> {fir.bindc_name = "c"}) {
! CHECK:           %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[VAL_0]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFsubstring_4Ec"} : (!fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>)
! CHECK:           %[[VAL_2:.*]] = fir.load %[[VAL_1]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>
! CHECK:           %[[VAL_3:.*]] = fir.box_addr %[[VAL_2]] : (!fir.box<!fir.ptr<!fir.char<1,?>>>) -> !fir.ptr<!fir.char<1,?>>
! CHECK:           omp.task depend(taskdependout -> %[[VAL_3]] : !fir.ptr<!fir.char<1,?>>) {
! CHECK:             omp.terminator
! CHECK:           }
! CHECK:           return
! CHECK:         }

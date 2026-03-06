! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=60 -o - %s | FileCheck %s


subroutine omp_interchange02(lb, ub, inc)
  integer res, i, lb, ub, inc

  !$omp interchange permutation(2,1)
  do i = lb, ub, inc
    do j = lb, ub, inc
      res = j
    end do
  end do
  !$omp end interchange

end subroutine omp_interchange02

! CHECK-LABEL:   func.func @_QPomp_interchange02(
! CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<i32> {fir.bindc_name = "lb"},
! CHECK-SAME:      %[[ARG1:.*]]: !fir.ref<i32> {fir.bindc_name = "ub"},
! CHECK-SAME:      %[[ARG2:.*]]: !fir.ref<i32> {fir.bindc_name = "inc"}) {
! CHECK:           %[[DUMMY_SCOPE_0:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[ALLOCA_0:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFomp_interchange02Ei"}
! CHECK:           %[[DECLARE_0:.*]]:2 = hlfir.declare %[[ALLOCA_0]] {uniq_name = "_QFomp_interchange02Ei"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[DECLARE_1:.*]]:2 = hlfir.declare %[[ARG2]] dummy_scope %[[DUMMY_SCOPE_0]] arg 3 {uniq_name = "_QFomp_interchange02Einc"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[ALLOCA_1:.*]] = fir.alloca i32 {bindc_name = "j", uniq_name = "_QFomp_interchange02Ej"}
! CHECK:           %[[DECLARE_2:.*]]:2 = hlfir.declare %[[ALLOCA_1]] {uniq_name = "_QFomp_interchange02Ej"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[DECLARE_3:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[DUMMY_SCOPE_0]] arg 1 {uniq_name = "_QFomp_interchange02Elb"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[ALLOCA_2:.*]] = fir.alloca i32 {bindc_name = "res", uniq_name = "_QFomp_interchange02Eres"}
! CHECK:           %[[DECLARE_4:.*]]:2 = hlfir.declare %[[ALLOCA_2]] {uniq_name = "_QFomp_interchange02Eres"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[DECLARE_5:.*]]:2 = hlfir.declare %[[ARG1]] dummy_scope %[[DUMMY_SCOPE_0]] arg 2 {uniq_name = "_QFomp_interchange02Eub"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[LOAD_0:.*]] = fir.load %[[DECLARE_3]]#0 : !fir.ref<i32>
! CHECK:           %[[LOAD_1:.*]] = fir.load %[[DECLARE_5]]#0 : !fir.ref<i32>
! CHECK:           %[[LOAD_2:.*]] = fir.load %[[DECLARE_1]]#0 : !fir.ref<i32>
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
! CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : i32
! CHECK:           %[[CMPI_0:.*]] = arith.cmpi slt, %[[LOAD_2]], %[[CONSTANT_0]] : i32
! CHECK:           %[[SUBI_0:.*]] = arith.subi %[[CONSTANT_0]], %[[LOAD_2]] : i32
! CHECK:           %[[SELECT_0:.*]] = arith.select %[[CMPI_0]], %[[SUBI_0]], %[[LOAD_2]] : i32
! CHECK:           %[[SELECT_1:.*]] = arith.select %[[CMPI_0]], %[[LOAD_1]], %[[LOAD_0]] : i32
! CHECK:           %[[SELECT_2:.*]] = arith.select %[[CMPI_0]], %[[LOAD_0]], %[[LOAD_1]] : i32
! CHECK:           %[[SUBI_1:.*]] = arith.subi %[[SELECT_2]], %[[SELECT_1]] overflow<nuw> : i32
! CHECK:           %[[DIVUI_0:.*]] = arith.divui %[[SUBI_1]], %[[SELECT_0]] : i32
! CHECK:           %[[ADDI_0:.*]] = arith.addi %[[DIVUI_0]], %[[CONSTANT_1]] overflow<nuw> : i32
! CHECK:           %[[CMPI_1:.*]] = arith.cmpi slt, %[[SELECT_2]], %[[SELECT_1]] : i32
! CHECK:           %[[SELECT_3:.*]] = arith.select %[[CMPI_1]], %[[CONSTANT_0]], %[[ADDI_0]] : i32
! CHECK:           %[[NEW_CLI_0:.*]] = omp.new_cli
! CHECK:           %[[LOAD_3:.*]] = fir.load %[[DECLARE_3]]#0 : !fir.ref<i32>
! CHECK:           %[[LOAD_4:.*]] = fir.load %[[DECLARE_5]]#0 : !fir.ref<i32>
! CHECK:           %[[LOAD_5:.*]] = fir.load %[[DECLARE_1]]#0 : !fir.ref<i32>
! CHECK:           %[[CONSTANT_2:.*]] = arith.constant 0 : i32
! CHECK:           %[[CONSTANT_3:.*]] = arith.constant 1 : i32
! CHECK:           %[[CMPI_2:.*]] = arith.cmpi slt, %[[LOAD_5]], %[[CONSTANT_2]] : i32
! CHECK:           %[[SUBI_2:.*]] = arith.subi %[[CONSTANT_2]], %[[LOAD_5]] : i32
! CHECK:           %[[SELECT_4:.*]] = arith.select %[[CMPI_2]], %[[SUBI_2]], %[[LOAD_5]] : i32
! CHECK:           %[[SELECT_5:.*]] = arith.select %[[CMPI_2]], %[[LOAD_4]], %[[LOAD_3]] : i32
! CHECK:           %[[SELECT_6:.*]] = arith.select %[[CMPI_2]], %[[LOAD_3]], %[[LOAD_4]] : i32
! CHECK:           %[[SUBI_3:.*]] = arith.subi %[[SELECT_6]], %[[SELECT_5]] overflow<nuw> : i32
! CHECK:           %[[DIVUI_1:.*]] = arith.divui %[[SUBI_3]], %[[SELECT_4]] : i32
! CHECK:           %[[ADDI_1:.*]] = arith.addi %[[DIVUI_1]], %[[CONSTANT_3]] overflow<nuw> : i32
! CHECK:           %[[CMPI_3:.*]] = arith.cmpi slt, %[[SELECT_6]], %[[SELECT_5]] : i32
! CHECK:           %[[SELECT_7:.*]] = arith.select %[[CMPI_3]], %[[CONSTANT_2]], %[[ADDI_1]] : i32
! CHECK:           %[[NEW_CLI_1:.*]] = omp.new_cli
! CHECK:           omp.canonical_loop(%[[NEW_CLI_0]]) %[[VAL_0:.*]] : i32 in range(%[[SELECT_3]]) {
! CHECK:             omp.canonical_loop(%[[NEW_CLI_1]]) %[[VAL_1:.*]] : i32 in range(%[[SELECT_7]]) {
! CHECK:               %[[MULI_0:.*]] = arith.muli %[[VAL_0]], %[[LOAD_2]] : i32
! CHECK:               %[[ADDI_2:.*]] = arith.addi %[[LOAD_0]], %[[MULI_0]] : i32
! CHECK:               hlfir.assign %[[ADDI_2]] to %[[DECLARE_0]]#0 : i32, !fir.ref<i32>
! CHECK:               %[[MULI_1:.*]] = arith.muli %[[VAL_1]], %[[LOAD_5]] : i32
! CHECK:               %[[ADDI_3:.*]] = arith.addi %[[LOAD_3]], %[[MULI_1]] : i32
! CHECK:               hlfir.assign %[[ADDI_3]] to %[[DECLARE_2]]#0 : i32, !fir.ref<i32>
! CHECK:               %[[LOAD_6:.*]] = fir.load %[[DECLARE_2]]#0 : !fir.ref<i32>
! CHECK:               hlfir.assign %[[LOAD_6]] to %[[DECLARE_4]]#0 : i32, !fir.ref<i32>
! CHECK:               omp.terminator
! CHECK:             }
! CHECK:             omp.terminator
! CHECK:           }
! CHECK:           %[[NEW_CLI_2:.*]] = omp.new_cli
! CHECK:           %[[NEW_CLI_3:.*]] = omp.new_cli
! CHECK:           omp.interchange (%[[NEW_CLI_2]], %[[NEW_CLI_3]]) <- (%[[NEW_CLI_0]], %[[NEW_CLI_1]]) permutation([2, 1])
! CHECK:           return
! CHECK:         }

! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=60 -o - %s | FileCheck %s


subroutine omp_fuse01(lb1, ub1, inc1, lb2, ub2, inc2)
  integer res, i, j
  integer lb1, ub1, inc1
  integer lb2, ub2, inc2

  !$omp fuse
  do i = lb1, ub1, inc1
    res = i
  end do
  do j = lb2, ub2, inc2
    res = j
  end do
  !$omp end fuse

end subroutine omp_fuse01


! CHECK-LABEL:   func.func @_QPomp_fuse01(
! CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<i32> {fir.bindc_name = "lb1"},
! CHECK-SAME:      %[[ARG1:.*]]: !fir.ref<i32> {fir.bindc_name = "ub1"},
! CHECK-SAME:      %[[ARG2:.*]]: !fir.ref<i32> {fir.bindc_name = "inc1"},
! CHECK-SAME:      %[[ARG3:.*]]: !fir.ref<i32> {fir.bindc_name = "lb2"},
! CHECK-SAME:      %[[ARG4:.*]]: !fir.ref<i32> {fir.bindc_name = "ub2"},
! CHECK-SAME:      %[[ARG5:.*]]: !fir.ref<i32> {fir.bindc_name = "inc2"}) {
! CHECK:           %[[DUMMY_SCOPE_0:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[ALLOCA_0:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFomp_fuse01Ei"}
! CHECK:           %[[DECLARE_0:.*]]:2 = hlfir.declare %[[ALLOCA_0]] {uniq_name = "_QFomp_fuse01Ei"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[DECLARE_1:.*]]:2 = hlfir.declare %[[ARG2]] dummy_scope %[[DUMMY_SCOPE_0]] arg {{[0-9]+}} {uniq_name = "_QFomp_fuse01Einc1"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[DECLARE_2:.*]]:2 = hlfir.declare %[[ARG5]] dummy_scope %[[DUMMY_SCOPE_0]] arg {{[0-9]+}} {uniq_name = "_QFomp_fuse01Einc2"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[ALLOCA_1:.*]] = fir.alloca i32 {bindc_name = "j", uniq_name = "_QFomp_fuse01Ej"}
! CHECK:           %[[DECLARE_3:.*]]:2 = hlfir.declare %[[ALLOCA_1]] {uniq_name = "_QFomp_fuse01Ej"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[DECLARE_4:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[DUMMY_SCOPE_0]] arg {{[0-9]+}} {uniq_name = "_QFomp_fuse01Elb1"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[DECLARE_5:.*]]:2 = hlfir.declare %[[ARG3]] dummy_scope %[[DUMMY_SCOPE_0]] arg {{[0-9]+}} {uniq_name = "_QFomp_fuse01Elb2"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[ALLOCA_2:.*]] = fir.alloca i32 {bindc_name = "res", uniq_name = "_QFomp_fuse01Eres"}
! CHECK:           %[[DECLARE_6:.*]]:2 = hlfir.declare %[[ALLOCA_2]] {uniq_name = "_QFomp_fuse01Eres"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[DECLARE_7:.*]]:2 = hlfir.declare %[[ARG1]] dummy_scope %[[DUMMY_SCOPE_0]] arg {{[0-9]+}} {uniq_name = "_QFomp_fuse01Eub1"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[DECLARE_8:.*]]:2 = hlfir.declare %[[ARG4]] dummy_scope %[[DUMMY_SCOPE_0]] arg {{[0-9]+}} {uniq_name = "_QFomp_fuse01Eub2"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[LOAD_0:.*]] = fir.load %[[DECLARE_4]]#0 : !fir.ref<i32>
! CHECK:           %[[LOAD_1:.*]] = fir.load %[[DECLARE_7]]#0 : !fir.ref<i32>
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
! CHECK:           omp.canonical_loop(%[[NEW_CLI_0]]) %[[VAL_0:.*]] : i32 in range(%[[SELECT_3]]) {
! CHECK:             %[[MULI_0:.*]] = arith.muli %[[VAL_0]], %[[LOAD_2]] : i32
! CHECK:             %[[ADDI_1:.*]] = arith.addi %[[LOAD_0]], %[[MULI_0]] : i32
! CHECK:             hlfir.assign %[[ADDI_1]] to %[[DECLARE_0]]#0 : i32, !fir.ref<i32>
! CHECK:             %[[LOAD_3:.*]] = fir.load %[[DECLARE_0]]#0 : !fir.ref<i32>
! CHECK:             hlfir.assign %[[LOAD_3]] to %[[DECLARE_6]]#0 : i32, !fir.ref<i32>
! CHECK:             omp.terminator
! CHECK:           }
! CHECK:           %[[LOAD_4:.*]] = fir.load %[[DECLARE_5]]#0 : !fir.ref<i32>
! CHECK:           %[[LOAD_5:.*]] = fir.load %[[DECLARE_8]]#0 : !fir.ref<i32>
! CHECK:           %[[LOAD_6:.*]] = fir.load %[[DECLARE_2]]#0 : !fir.ref<i32>
! CHECK:           %[[CONSTANT_2:.*]] = arith.constant 0 : i32
! CHECK:           %[[CONSTANT_3:.*]] = arith.constant 1 : i32
! CHECK:           %[[CMPI_2:.*]] = arith.cmpi slt, %[[LOAD_6]], %[[CONSTANT_2]] : i32
! CHECK:           %[[SUBI_2:.*]] = arith.subi %[[CONSTANT_2]], %[[LOAD_6]] : i32
! CHECK:           %[[SELECT_4:.*]] = arith.select %[[CMPI_2]], %[[SUBI_2]], %[[LOAD_6]] : i32
! CHECK:           %[[SELECT_5:.*]] = arith.select %[[CMPI_2]], %[[LOAD_5]], %[[LOAD_4]] : i32
! CHECK:           %[[SELECT_6:.*]] = arith.select %[[CMPI_2]], %[[LOAD_4]], %[[LOAD_5]] : i32
! CHECK:           %[[SUBI_3:.*]] = arith.subi %[[SELECT_6]], %[[SELECT_5]] overflow<nuw> : i32
! CHECK:           %[[DIVUI_1:.*]] = arith.divui %[[SUBI_3]], %[[SELECT_4]] : i32
! CHECK:           %[[ADDI_2:.*]] = arith.addi %[[DIVUI_1]], %[[CONSTANT_3]] overflow<nuw> : i32
! CHECK:           %[[CMPI_3:.*]] = arith.cmpi slt, %[[SELECT_6]], %[[SELECT_5]] : i32
! CHECK:           %[[SELECT_7:.*]] = arith.select %[[CMPI_3]], %[[CONSTANT_2]], %[[ADDI_2]] : i32
! CHECK:           %[[NEW_CLI_1:.*]] = omp.new_cli
! CHECK:           omp.canonical_loop(%[[NEW_CLI_1]]) %[[VAL_1:.*]] : i32 in range(%[[SELECT_7]]) {
! CHECK:             %[[MULI_1:.*]] = arith.muli %[[VAL_1]], %[[LOAD_6]] : i32
! CHECK:             %[[ADDI_3:.*]] = arith.addi %[[LOAD_4]], %[[MULI_1]] : i32
! CHECK:             hlfir.assign %[[ADDI_3]] to %[[DECLARE_3]]#0 : i32, !fir.ref<i32>
! CHECK:             %[[LOAD_7:.*]] = fir.load %[[DECLARE_3]]#0 : !fir.ref<i32>
! CHECK:             hlfir.assign %[[LOAD_7]] to %[[DECLARE_6]]#0 : i32, !fir.ref<i32>
! CHECK:             omp.terminator
! CHECK:           }
! CHECK:           %[[NEW_CLI_2:.*]] = omp.new_cli
! CHECK:           omp.fuse (%[[NEW_CLI_2]]) <- (%[[NEW_CLI_0]], %[[NEW_CLI_1]])
! CHECK:           return
! CHECK:         }


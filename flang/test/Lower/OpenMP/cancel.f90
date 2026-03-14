! RUN: %flang_fc1 -fopenmp -emit-hlfir %s -o - | FileCheck %s
! RUN: bbc -fopenmp -emit-hlfir %s -o - | FileCheck %s

subroutine cancel_parallel()
  !$omp parallel
    !$omp cancel parallel
  !$omp end parallel
end subroutine
! CHECK-LABEL:   func.func @_QPcancel_parallel() {
! CHECK:           omp.parallel {
! CHECK:             omp.cancel cancellation_construct_type(parallel)
! CHECK:             omp.terminator
! CHECK:           }
! CHECK:           return
! CHECK:         }

subroutine cancel_do()
  !$omp parallel do
  do i = 1,100
    !$omp cancel do
  enddo
  !$omp end parallel do
end subroutine
! CHECK-LABEL:   func.func @_QPcancel_do() {
! CHECK:           %[[VAL_0:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFcancel_doEi"}
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFcancel_doEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           omp.parallel {
! CHECK:             %[[VAL_2:.*]] = arith.constant 1 : i32
! CHECK:             %[[VAL_3:.*]] = arith.constant 100 : i32
! CHECK:             %[[VAL_4:.*]] = arith.constant 1 : i32
! CHECK:             omp.wsloop private(@_QFcancel_doEi_private_i32 %[[VAL_1]]#0 -> %[[VAL_5:.*]] : !fir.ref<i32>) {
! CHECK:               omp.loop_nest (%[[VAL_6:.*]]) : i32 = (%[[VAL_2]]) to (%[[VAL_3]]) inclusive step (%[[VAL_4]]) {
! CHECK:                 %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_5]] {uniq_name = "_QFcancel_doEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:                 hlfir.assign %[[VAL_6]] to %[[VAL_7]]#0 : i32, !fir.ref<i32>
! CHECK:                 omp.cancel cancellation_construct_type(loop)
! CHECK:                 omp.yield
! CHECK:               }
! CHECK:             }
! CHECK:             omp.terminator
! CHECK:           }
! CHECK:           return
! CHECK:         }

subroutine cancel_sections()
  !$omp sections
    !$omp section
      !$omp cancel sections
  !$omp end sections
end subroutine
! CHECK-LABEL:   func.func @_QPcancel_sections() {
! CHECK:           omp.sections {
! CHECK:             omp.section {
! CHECK:               omp.cancel cancellation_construct_type(sections)
! CHECK:               omp.terminator
! CHECK:             }
! CHECK:             omp.terminator
! CHECK:           }
! CHECK:           return
! CHECK:         }

subroutine cancel_taskgroup()
  !$omp taskgroup
    !$omp task
      !$omp cancel taskgroup
    !$omp end task
  !$omp end taskgroup
end subroutine
! CHECK-LABEL:   func.func @_QPcancel_taskgroup() {
! CHECK:           omp.taskgroup {
! CHECK:             omp.task {
! CHECK:               omp.cancel cancellation_construct_type(taskgroup)
! CHECK:               omp.terminator
! CHECK:             }
! CHECK:             omp.terminator
! CHECK:           }
! CHECK:           return
! CHECK:         }

subroutine cancel_parallel_if(cond)
  logical :: cond
  !$omp parallel
    !$omp cancel parallel if(cond)
  !$omp end parallel
end subroutine
! CHECK-LABEL:   func.func @_QPcancel_parallel_if(
! CHECK-SAME:                                     %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.ref<!fir.logical<4>> {fir.bindc_name = "cond"}) {
! CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_1]] arg {{[0-9]+}} {uniq_name = "_QFcancel_parallel_ifEcond"} : (!fir.ref<!fir.logical<4>>, !fir.dscope) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
! CHECK:           omp.parallel {
! CHECK:             %[[VAL_3:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<!fir.logical<4>>
! CHECK:             %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (!fir.logical<4>) -> i1
! CHECK:             omp.cancel cancellation_construct_type(parallel) if(%[[VAL_4]])
! CHECK:             omp.terminator
! CHECK:           }
! CHECK:           return
! CHECK:         }

subroutine cancel_do_if(cond)
  logical :: cond
  !$omp parallel do
  do i = 1,100
    !$omp cancel do if (cond)
  enddo
  !$omp end parallel do
end subroutine
! CHECK-LABEL:   func.func @_QPcancel_do_if(
! CHECK-SAME:                               %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.ref<!fir.logical<4>> {fir.bindc_name = "cond"}) {
! CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_1]] arg {{[0-9]+}} {uniq_name = "_QFcancel_do_ifEcond"} : (!fir.ref<!fir.logical<4>>, !fir.dscope) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
! CHECK:           %[[VAL_3:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFcancel_do_ifEi"}
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_3]] {uniq_name = "_QFcancel_do_ifEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           omp.parallel {
! CHECK:             %[[VAL_5:.*]] = arith.constant 1 : i32
! CHECK:             %[[VAL_6:.*]] = arith.constant 100 : i32
! CHECK:             %[[VAL_7:.*]] = arith.constant 1 : i32
! CHECK:             omp.wsloop private(@_QFcancel_do_ifEi_private_i32 %[[VAL_4]]#0 -> %[[VAL_8:.*]] : !fir.ref<i32>) {
! CHECK:               omp.loop_nest (%[[VAL_9:.*]]) : i32 = (%[[VAL_5]]) to (%[[VAL_6]]) inclusive step (%[[VAL_7]]) {
! CHECK:                 %[[VAL_10:.*]]:2 = hlfir.declare %[[VAL_8]] {uniq_name = "_QFcancel_do_ifEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:                 hlfir.assign %[[VAL_9]] to %[[VAL_10]]#0 : i32, !fir.ref<i32>
! CHECK:                 %[[VAL_11:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<!fir.logical<4>>
! CHECK:                 %[[VAL_12:.*]] = fir.convert %[[VAL_11]] : (!fir.logical<4>) -> i1
! CHECK:                 omp.cancel cancellation_construct_type(loop) if(%[[VAL_12]])
! CHECK:                 omp.yield
! CHECK:               }
! CHECK:             }
! CHECK:             omp.terminator
! CHECK:           }
! CHECK:           return
! CHECK:         }

subroutine cancel_sections_if(cond)
  logical :: cond
  !$omp sections
    !$omp section
    !$omp cancel sections if(cond)
  !$omp end sections
end subroutine
! CHECK-LABEL:   func.func @_QPcancel_sections_if(
! CHECK-SAME:                                     %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.ref<!fir.logical<4>> {fir.bindc_name = "cond"}) {
! CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_1]] arg {{[0-9]+}} {uniq_name = "_QFcancel_sections_ifEcond"} : (!fir.ref<!fir.logical<4>>, !fir.dscope) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
! CHECK:           omp.sections {
! CHECK:             omp.section {
! CHECK:               %[[VAL_3:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<!fir.logical<4>>
! CHECK:               %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (!fir.logical<4>) -> i1
! CHECK:               omp.cancel cancellation_construct_type(sections) if(%[[VAL_4]])
! CHECK:               omp.terminator
! CHECK:             }
! CHECK:             omp.terminator
! CHECK:           }
! CHECK:           return
! CHECK:         }

subroutine cancel_taskgroup_if(cond)
  logical :: cond
  !$omp taskgroup
    !$omp task
      !$omp cancel taskgroup if(cond)
    !$omp end task
  !$omp end taskgroup
end subroutine
! CHECK-LABEL:   func.func @_QPcancel_taskgroup_if(
! CHECK-SAME:                                      %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.ref<!fir.logical<4>> {fir.bindc_name = "cond"}) {
! CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_1]] arg {{[0-9]+}} {uniq_name = "_QFcancel_taskgroup_ifEcond"} : (!fir.ref<!fir.logical<4>>, !fir.dscope) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
! CHECK:           omp.taskgroup {
! CHECK:             omp.task {
! CHECK:               %[[VAL_3:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<!fir.logical<4>>
! CHECK:               %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (!fir.logical<4>) -> i1
! CHECK:               omp.cancel cancellation_construct_type(taskgroup) if(%[[VAL_4]])
! CHECK:               omp.terminator
! CHECK:             }
! CHECK:             omp.terminator
! CHECK:           }
! CHECK:           return
! CHECK:         }

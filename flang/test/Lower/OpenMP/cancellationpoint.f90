! RUN: %flang_fc1 -fopenmp -emit-hlfir %s -o - | FileCheck %s
! RUN: bbc -fopenmp -emit-hlfir %s -o - | FileCheck %s

subroutine cancellationpoint_parallel()
  !$omp parallel
    !$omp cancellationpoint parallel
  !$omp end parallel
end subroutine
! CHECK-LABEL:   func.func @_QPcancellationpoint_parallel() {
! CHECK:           omp.parallel {
! CHECK:             omp.cancellation_point cancellation_construct_type(parallel)
! CHECK:             omp.terminator
! CHECK:           }
! CHECK:           return
! CHECK:         }

subroutine cancellationpoint_do()
  !$omp parallel do
  do i = 1,100
    !$omp cancellationpoint do
  enddo
  !$omp end parallel do
end subroutine
! CHECK-LABEL:   func.func @_QPcancellationpoint_do() {
! CHECK:           %[[VAL_0:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFcancellationpoint_doEi"}
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFcancellationpoint_doEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           omp.parallel {
! CHECK:             %[[VAL_2:.*]] = arith.constant 1 : i32
! CHECK:             %[[VAL_3:.*]] = arith.constant 100 : i32
! CHECK:             %[[VAL_4:.*]] = arith.constant 1 : i32
! CHECK:             omp.wsloop private(@_QFcancellationpoint_doEi_private_i32 %[[VAL_1]]#0 -> %[[VAL_5:.*]] : !fir.ref<i32>) {
! CHECK:               omp.loop_nest (%[[VAL_6:.*]]) : i32 = (%[[VAL_2]]) to (%[[VAL_3]]) inclusive step (%[[VAL_4]]) {
! CHECK:                 %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_5]] {uniq_name = "_QFcancellationpoint_doEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:                 hlfir.assign %[[VAL_6]] to %[[VAL_7]]#0 : i32, !fir.ref<i32>
! CHECK:                 omp.cancellation_point cancellation_construct_type(loop)
! CHECK:                 omp.yield
! CHECK:               }
! CHECK:             }
! CHECK:             omp.terminator
! CHECK:           }
! CHECK:           return
! CHECK:         }

subroutine cancellationpoint_sections()
  !$omp sections
    !$omp section
      !$omp cancellationpoint sections
  !$omp end sections
end subroutine
! CHECK-LABEL:   func.func @_QPcancellationpoint_sections() {
! CHECK:           omp.sections {
! CHECK:             omp.section {
! CHECK:               omp.cancellation_point cancellation_construct_type(sections)
! CHECK:               omp.terminator
! CHECK:             }
! CHECK:             omp.terminator
! CHECK:           }
! CHECK:           return
! CHECK:         }

subroutine cancellationpoint_taskgroup()
  !$omp taskgroup
    !$omp task
      !$omp cancellationpoint taskgroup
    !$omp end task
  !$omp end taskgroup
end subroutine
! CHECK-LABEL:   func.func @_QPcancellationpoint_taskgroup() {
! CHECK:           omp.taskgroup {
! CHECK:             omp.task {
! CHECK:               omp.cancellation_point cancellation_construct_type(taskgroup)
! CHECK:               omp.terminator
! CHECK:             }
! CHECK:             omp.terminator
! CHECK:           }
! CHECK:           return
! CHECK:         }

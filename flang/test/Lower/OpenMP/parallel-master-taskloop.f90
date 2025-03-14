! This test checks lowering of OpenMP parallel masked taskloop Directive.

! RUN: bbc -emit-hlfir -fopenmp -fopenmp-version=50 -o - %s 2>&1 | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 -o - %s 2>&1 | FileCheck %s

! CHECK: func.func @_QPtest_parallel_master_taskloop() {
! CHECK:    %[[VAL_0:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtest_parallel_master_taskloopEi"}
! CHECK:    %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFtest_parallel_master_taskloopEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:    %[[VAL_2:.*]] = fir.address_of(@_QFtest_parallel_master_taskloopEj) : !fir.ref<i32>
! CHECK:    %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]] {uniq_name = "_QFtest_parallel_master_taskloopEj"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:    omp.parallel {
! CHECK:      omp.masked {
! CHECK:        %[[VAL_4:.*]] = fir.alloca i32 {bindc_name = "i", pinned, uniq_name = "_QFtest_parallel_master_taskloopEi"}
! CHECK:        %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_4]] {uniq_name = "_QFtest_parallel_master_taskloopEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:        %[[C1_I32:.*]] = arith.constant 1 : i32
! CHECK:        %[[C10_I32:.*]] = arith.constant 10 : i32
! CHECK:        %[[C1_I32_0:.*]] = arith.constant 1 : i32
! CHECK:        omp.taskloop {
! CHECK:          omp.loop_nest (%arg0) : i32 = (%[[C1_I32]]) to (%[[C10_I32]]) inclusive step (%[[C1_I32_0]]) {
! CHECK:            fir.store %arg0 to %[[VAL_5]]#1 : !fir.ref<i32>
! CHECK:            %[[VAL_6:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<i32>
! CHECK:            %[[C1_I32_1:.*]] = arith.constant 1 : i32
! CHECK:            %[[VAL_7:.*]] = arith.addi %[[VAL_6]], %[[C1_I32_1]] : i32
! CHECK:            hlfir.assign %[[VAL_7]] to %[[VAL_3]]#0 : i32, !fir.ref<i32>
! CHECK:            omp.yield
! CHECK:          }
! CHECK:        }
! CHECK:        omp.terminator
! CHECK:      }
! CHECK:      omp.terminator
! CHECK:    }
! CHECK:    return
! CHECK:  }
! CHECK:  fir.global internal @_QFtest_parallel_master_taskloopEj : i32 {
! CHECK:    %[[C1_I32:.*]] = arith.constant 1 : i32
! CHECK:    fir.has_value %[[C1_I32]] : i32
! CHECK:  }

subroutine test_parallel_master_taskloop
  integer :: i, j = 1
  ! OpenMP directive PARALLEL MASTER TASKLOOP has been deprecated, so used PARALLEL MASKED TASKLOOP instead.
  !$omp parallel masked taskloop
  do i=1,10
   j = j + 1
  end do
  !$omp end parallel masked taskloop 
end subroutine

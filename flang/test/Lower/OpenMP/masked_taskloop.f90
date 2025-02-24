! This test checks lowering of OpenMP masked taskloop Directive.

! RUN:  bbc -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s
! RUN:  %flang_fc1 -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s

! CHECK:  func.func @_QPtest_masked_taskloop() {
! CHECK:    %[[VAL_0:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtest_masked_taskloopEi"}
! CHECK:    %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFtest_masked_taskloopEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:    %[[VAL_2:.*]] = fir.address_of(@_QFtest_masked_taskloopEj) : !fir.ref<i32>
! CHECK:    %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]] {uniq_name = "_QFtest_masked_taskloopEj"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:    omp.masked {
! CHECK:      %[[VAL_4:.*]] = fir.alloca i32 {bindc_name = "j", pinned, uniq_name = "_QFtest_masked_taskloopEj"}
! CHECK:      %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_4]] {uniq_name = "_QFtest_masked_taskloopEj"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:      %[[VAL_6:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<i32>
! CHECK:      hlfir.assign %[[VAL_6]] to %[[VAL_5]]#0 : i32, !fir.ref<i32>
! CHECK:      %[[VAL_7:.*]] = fir.alloca i32 {bindc_name = "i", pinned, uniq_name = "_QFtest_masked_taskloopEi"}
! CHECK:      %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_7]] {uniq_name = "_QFtest_masked_taskloopEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:      %[[C1_I32:.*]] = arith.constant 1 : i32
! CHECK:      %[[C10_I32:.*]] = arith.constant 10 : i32
! CHECK:      %[[C1_I32_0:.*]] = arith.constant 1 : i32
! CHECK:      omp.taskloop {
! CHECK:        omp.loop_nest (%arg0) : i32 = (%[[C1_I32]]) to (%[[C10_I32]]) inclusive step (%[[C1_I32_0]]) {
! CHECK:          fir.store %arg0 to %[[VAL_8]]#1 : !fir.ref<i32>
! CHECK:          %[[VAL_9:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<i32>
! CHECK:          %[[C1_I32_1:.*]] = arith.constant 1 : i32
! CHECK:          %[[VAL_10:.*]] = arith.addi %[[VAL_9]], %[[C1_I32_1]] : i32
! CHECK:          hlfir.assign %[[VAL_10]] to %[[VAL_5]]#0 : i32, !fir.ref<i32>
! CHECK:          omp.yield
! CHECK:        }
! CHECK:      }
! CHECK:      omp.terminator
! CHECK:    }
! CHECK:    return
! CHECK:  }
! CHECK:  fir.global internal @_QFtest_masked_taskloopEj : i32 {
! CHECK:    %[[C1_I32:.*]] = arith.constant 1 : i32
! CHECK:    fir.has_value %[[C1_I32]] : i32
! CHECK: }


subroutine test_masked_taskloop
  integer :: i, j = 1
  !OpenMP directive MASKED has been deprecated, so used MASKED TASKLOOP instead.
  !$omp masked taskloop
  do i=1,10
   j = j + 1
  end do
  !$omp end masked taskloop 
end subroutine

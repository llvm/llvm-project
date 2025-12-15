! Test the collapse clause when being used with the taskloop construct
! RUN: bbc -emit-hlfir -fopenmp -fopenmp-version=45 %s -o - 2>&1 | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=45 %s -o - 2>&1 | FileCheck %s

! CHECK-LABEL: omp.private
! CHECK-SAME: {type = private} @[[J_PRIVATE:.*]] : i32
! CHECK-LABEL: omp.private
! CHECK-SAME: {type = private} @[[I_PRIVATE:.*]] : i32
! CHECK-LABEL: omp.private
! CHECK-SAME: {type = firstprivate} @[[SUM_FIRSTPRIVATE:.*]] : i32 copy

! CHECK-LABEL: func.func @_QPtest()
! CHECK: %[[ALLOCA_I:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtestEi"}
! CHECK: %[[DECLARE_I:.*]]:2 = hlfir.declare %1 {uniq_name = "_QFtestEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK: %[[ALLOCA_J:.*]] = fir.alloca i32 {bindc_name = "j", uniq_name = "_QFtestEj"}
! CHECK: %[[DECLARE_J:.*]]:2 = hlfir.declare %3 {uniq_name = "_QFtestEj"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK: %[[ALLOCA_SUM:.*]] = fir.alloca i32 {bindc_name = "sum", uniq_name = "_QFtestEsum"}
! CHECK: %[[DECLARE_SUM:.*]]:2 = hlfir.declare %5 {uniq_name = "_QFtestEsum"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

subroutine test()
    integer :: i, j, sum

    !$omp taskloop collapse(2)
    ! CHECK-LABEL: omp.taskloop
    ! CHECK-SAME: private(@_QFtestEsum_firstprivate_i32 %[[DECLARE_SUM]]#0 -> %arg0, @_QFtestEi_private_i32 %[[DECLARE_I]]#0 -> %arg1, @_QFtestEj_private_i32 %[[DECLARE_J]]#0 -> %arg2 : !fir.ref<i32>, !fir.ref<i32>, !fir.ref<i32>)
    ! CHECK-LABEL: omp.loop_nest
    ! CHECK-SAME: (%arg3, %arg4) : i32 = (%c1_i32, %c1_i32_1) to (%c10_i32, %c5_i32) inclusive step (%c1_i32_0, %c1_i32_2) collapse(2)
    do i = 1, 10
        do j = 1, 5
            sum = sum + i + j
        end do
    end do
    !$omp end taskloop
end subroutine

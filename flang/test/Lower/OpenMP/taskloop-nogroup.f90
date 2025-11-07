! Test the Nogroup clause when used with the taskloop directive
! RUN: bbc -emit-hlfir -fopenmp -fopenmp-version=45 %s -o - 2>&1 | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=45 %s -o - 2>&1 | FileCheck %s

! CHECK-LABEL: omp.private
! CHECK-SAME: {type = private} @[[I_PRIVATE:.*]] : i32
! CHECK-LABEL: omp.private
! CHECK-SAME: {type = firstprivate} @[[SUM_FIRSTPRIVATE:.*]] : i32 copy

! CHECK: %[[ALLOCA_I:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtestEi"}
! CHECK: %[[DECLARE_I:.*]]:2 = hlfir.declare %[[ALLOCA_I]] {uniq_name = "_QFtestEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK: %[[ALLOCA_SUM:.*]] = fir.alloca i32 {bindc_name = "sum", uniq_name = "_QFtestEsum"}
! CHECK: %[[DECLARE_SUM:.*]]:2 = hlfir.declare %[[ALLOCA_SUM]] {uniq_name = "_QFtestEsum"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

subroutine test()
    integer :: i, sum

    ! CHECK-LABEL: omp.taskloop
    ! CHECK-SAME: nogroup private(@_QFtestEsum_firstprivate_i32 %[[DECLARE_SUM]]#0 -> %arg0, @_QFtestEi_private_i32 %[[DECLARE_I]]#0 -> %arg1 : !fir.ref<i32>, !fir.ref<i32>)
    !$omp taskloop nogroup
    do i=1,10
        sum = sum + i
    end do
    !$omp end taskloop
end subroutine

! This test checks the lowering of the reduction clause in the taskloop construct
! RUN: bbc -emit-hlfir -fopenmp -fopenmp-version=50 -o - %s 2>&1 | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 -o - %s 2>&1 | FileCheck %s
! RUN %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=45 -o - %s 2>&1 | FileCheck %s --check-prefix=CHECK-VERSION

! CHECK-VERSION: error: REDUCTION clause is not allowed on directive TASKLOOP in OpenMP v4.5, try -fopenmp-version=50

! CHECK-LABEL: omp.private
! CHECK-SAME: {type = private} @[[I_PRIVATE:.*]] : i32

! CHECK-LABEL: func.func @_QPtest_reduction()
! CHECK: %[[ALLOCA_A:.*]] = fir.alloca !fir.array<10xi32> {bindc_name = "a", uniq_name = "_QFtest_reductionEa"}
! CHECK: %[[DECLARE_A:.*]]:2 = hlfir.declare %[[ALLOCA_A]](%2) {uniq_name = "_QFtest_reductionEa"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
! CHECK: %[[ALLOCA_I:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtest_reductionEi"}
! CHECK: %[[DECLARE_I:.*]]:2 = hlfir.declare %[[ALLOCA_I]] {uniq_name = "_QFtest_reductionEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK: %[[ALLOCA_SUM_I:.*]] = fir.alloca i32 {bindc_name = "sum_i", uniq_name = "_QFtest_reductionEsum_i"}
! CHECK: %[[DECLARE_SUM_I:.*]]:2 = hlfir.declare %[[ALLOCA_SUM_I]] {uniq_name = "_QFtest_reductionEsum_i"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

subroutine test_reduction
    integer :: i, a(10), sum_i

    ! CHECK: omp.taskloop
    ! CHECK-SAME: private(@[[I_PRIVATE]] %[[DECLARE_I]]#0 -> %arg0 : !fir.ref<i32>) reduction(@add_reduction_i32 %[[DECLARE_SUM_I]]#0 -> %arg1 : !fir.ref<i32>) {
    !$omp taskloop reduction (+:sum_i)
    do i = 1,10
        sum_i = sum_i + i
    end do
    !$omp end taskloop

end subroutine
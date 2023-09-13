! RUN: bbc -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s

!CHECK: omp.reduction.declare @[[MAX_DECLARE_I:.*]] : i32 init {
!CHECK:   %[[MINIMUM_VAL_I:.*]] = arith.constant -2147483648 : i32
!CHECK:   omp.yield(%[[MINIMUM_VAL_I]] : i32)
!CHECK: combiner
!CHECK: ^bb0(%[[ARG0_I:.*]]: i32, %[[ARG1_I:.*]]: i32):
!CHECK:   %[[COMB_VAL_I:.*]] = arith.maxsi %[[ARG0_I]], %[[ARG1_I]] : i32
!CHECK:   omp.yield(%[[COMB_VAL_I]] : i32)

!CHECK-LABEL: @_QPreduction_max_int
!CHECK-SAME: %[[Y_BOX:.*]]: !fir.box<!fir.array<?xi32>>
!CHECK:   %[[X_REF:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFreduction_max_intEx"}
!CHECK:   %[[X_DECL:.*]]:2 = hlfir.declare %[[X_REF]] {uniq_name = "_QFreduction_max_intEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:   %[[Y_DECL:.*]]:2 = hlfir.declare %[[Y_BOX]] {uniq_name = "_QFreduction_max_intEy"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
!CHECK:   omp.parallel
!CHECK:     omp.wsloop reduction(@[[MAX_DECLARE_I]] -> %[[X_DECL]]#0 : !fir.ref<i32>) for
!CHECK:       %[[Y_I_REF:.*]] = hlfir.designate %[[Y_DECL]]#0 ({{.*}}) : (!fir.box<!fir.array<?xi32>>, i64) -> !fir.ref<i32>
!CHECK:       %[[Y_I:.*]] = fir.load %[[Y_I_REF]] : !fir.ref<i32>
!CHECK:       omp.reduction %[[Y_I]], %[[X_DECL]]#0 : i32, !fir.ref<i32>
!CHECK:       omp.yield
!CHECK:     omp.terminator

subroutine reduction_max_int(y)
  integer :: x, y(:)
  x = 0
  !$omp parallel
  !$omp do reduction(max:x)
  do i=1, 100
    x = max(x, y(i))
  end do
  !$omp end do
  !$omp end parallel
  print *, x
end subroutine

! RUN: bbc -emit-fir -fopenmp %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir -fopenmp %s -o - | FileCheck %s

!CHECK: omp.reduction.declare @[[IEOR_DECLARE_I:.*]] : i32 init {
!CHECK: %[[ZERO_VAL_I:.*]] = arith.constant 0 : i32
!CHECK: omp.yield(%[[ZERO_VAL_I]] : i32)
!CHECK: combiner
!CHECK: ^bb0(%[[ARG0_I:.*]]: i32, %[[ARG1_I:.*]]: i32):
!CHECK: %[[IEOR_VAL_I:.*]] = arith.xori %[[ARG0_I]], %[[ARG1_I]] : i32
!CHECK: omp.yield(%[[IEOR_VAL_I]] : i32)

!CHECK-LABEL: @_QPreduction_ieor
!CHECK-SAME: %[[Y_BOX:.*]]: !fir.box<!fir.array<?xi32>> 
!CHECK: %[[X_REF:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFreduction_ieorEx"}
!CHECK: omp.parallel
!CHECK: omp.wsloop reduction(@[[IEOR_DECLARE_I]] -> %[[X_REF]] : !fir.ref<i32>) for 
!CHECK: %[[Y_I_REF:.*]] = fir.coordinate_of %[[Y_BOX]]
!CHECK: %[[Y_I:.*]] = fir.load %[[Y_I_REF]] : !fir.ref<i32>
!CHECK: omp.reduction %[[Y_I]], %[[X_REF]] : i32, !fir.ref<i32>
!CHECK: omp.yield
!CHECK: omp.terminator

subroutine reduction_ieor(y)
  integer :: x, y(:)
  x = 0
  !$omp parallel
  !$omp do reduction(ieor:x)
  do i=1, 100
    x = ieor(x, y(i))
  end do
  !$omp end do
  !$omp end parallel
  print *, x
end subroutine

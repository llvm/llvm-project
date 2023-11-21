! RUN: bbc -emit-hlfir -fopenmp %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

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
!CHECK: %[[X_DECL:.*]]:2 = hlfir.declare %[[X_REF]] {uniq_name = "_QFreduction_ieorEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[Y_DECL:.*]]:2 = hlfir.declare %[[Y_BOX]] {uniq_name = "_QFreduction_ieorEy"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)


!CHECK: omp.parallel
!CHECK: %[[I_REF:.*]] = fir.alloca i32 {adapt.valuebyref, pinned}
!CHECK: %[[I_DECL:.*]]:2 = hlfir.declare %[[I_REF]] {uniq_name = "_QFreduction_ieorEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: omp.wsloop reduction(@[[IEOR_DECLARE_I]] -> %[[X_DECL]]#0 : !fir.ref<i32>) for 
!CHECK: fir.store %{{.*}} to %[[I_DECL]]#1 : !fir.ref<i32>
!CHECK: %[[I_32:.*]] = fir.load %[[I_DECL]]#0 : !fir.ref<i32>
!CHECK: %[[I_64:.*]] = fir.convert %[[I_32]] : (i32) -> i64
!CHECK: %[[Y_I_REF:.*]] = hlfir.designate %[[Y_DECL]]#0 (%[[I_64]])  : (!fir.box<!fir.array<?xi32>>, i64) -> !fir.ref<i32>
!CHECK: %[[Y_I:.*]] = fir.load %[[Y_I_REF]] : !fir.ref<i32>
!CHECK: omp.reduction %[[Y_I]], %[[X_DECL]]#0 : i32, !fir.ref<i32>
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

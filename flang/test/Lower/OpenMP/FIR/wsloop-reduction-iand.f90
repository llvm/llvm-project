! RUN: bbc -emit-fir -hlfir=false -fopenmp %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir -flang-deprecated-no-hlfir -fopenmp %s -o - | FileCheck %s

!CHECK: omp.reduction.declare @[[IAND_DECLARE_I:.*]] : i32 init {
!CHECK: %[[ZERO_VAL_I:.*]] = arith.constant -1 : i32
!CHECK: omp.yield(%[[ZERO_VAL_I]] : i32)
!CHECK: combiner
!CHECK: ^bb0(%[[ARG0_I:.*]]: i32, %[[ARG1_I:.*]]: i32):
!CHECK: %[[IAND_VAL_I:.*]] = arith.andi %[[ARG0_I]], %[[ARG1_I]] : i32
!CHECK: omp.yield(%[[IAND_VAL_I]] : i32)

!CHECK-LABEL: @_QPreduction_iand
!CHECK-SAME: %[[Y_BOX:.*]]: !fir.box<!fir.array<?xi32>>
!CHECK: %[[X_REF:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFreduction_iandEx"}
!CHECK: omp.parallel
!CHECK: omp.wsloop reduction(@[[IAND_DECLARE_I]] %[[X_REF]] -> %[[PRV:.+]] : !fir.ref<i32>) for
!CHECK: %[[LPRV:.+]] = fir.load %[[PRV]] : !fir.ref<i32>
!CHECK: %[[Y_I_REF:.*]] = fir.coordinate_of %[[Y_BOX]]
!CHECK: %[[Y_I:.*]] = fir.load %[[Y_I_REF]] : !fir.ref<i32>
!CHECK: %[[RES:.+]] = arith.andi %[[LPRV]], %[[Y_I]] : i32
!CHECK: fir.store %[[RES]] to %[[PRV]] : !fir.ref<i32>
!CHECK: omp.yield
!CHECK: omp.terminator

subroutine reduction_iand(y)
  integer :: x, y(:)
  x = 0
  !$omp parallel
  !$omp do reduction(iand:x)
  do i=1, 100
    x = iand(x, y(i))
  end do
  !$omp end do
  !$omp end parallel
  print *, x
end subroutine

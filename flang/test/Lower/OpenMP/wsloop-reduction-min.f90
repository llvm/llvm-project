! RUN: bbc -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s

!CHECK: omp.reduction.declare @[[MIN_DECLARE_F:.*]] : f32 init {
!CHECK:   %[[MAXIMUM_VAL_F:.*]] = arith.constant 3.40282347E+38 : f32
!CHECK:   omp.yield(%[[MAXIMUM_VAL_F]] : f32)
!CHECK: combiner
!CHECK: ^bb0(%[[ARG0_F:.*]]: f32, %[[ARG1_F:.*]]: f32):
!CHECK:   %[[COMB_VAL_F:.*]] = arith.minimumf %[[ARG0_F]], %[[ARG1_F]] {{.*}}: f32
!CHECK:   omp.yield(%[[COMB_VAL_F]] : f32)

!CHECK: omp.reduction.declare @[[MIN_DECLARE_I:.*]] : i32 init {
!CHECK:   %[[MAXIMUM_VAL_I:.*]] = arith.constant 2147483647 : i32
!CHECK:   omp.yield(%[[MAXIMUM_VAL_I]] : i32)
!CHECK: combiner
!CHECK: ^bb0(%[[ARG0_I:.*]]: i32, %[[ARG1_I:.*]]: i32):
!CHECK:   %[[COMB_VAL_I:.*]] = arith.minsi %[[ARG0_I]], %[[ARG1_I]] : i32
!CHECK:   omp.yield(%[[COMB_VAL_I]] : i32)

!CHECK-LABEL: @_QPreduction_min_int
!CHECK-SAME: %[[Y_BOX:.*]]: !fir.box<!fir.array<?xi32>>
!CHECK:   %[[X_REF:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFreduction_min_intEx"}
!CHECK:   %[[X_DECL:.*]]:2 = hlfir.declare %[[X_REF]] {uniq_name = "_QFreduction_min_intEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:   %[[Y_DECL:.*]]:2 = hlfir.declare %[[Y_BOX]] {uniq_name = "_QFreduction_min_intEy"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
!CHECK:   omp.parallel
!CHECK:      %[[I_REF:.*]] = fir.alloca i32 {adapt.valuebyref, pinned}
!CHECK:      %[[I_DECL:.*]]:2 = hlfir.declare %[[I_REF]] {uniq_name = "_QFreduction_min_intEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:     omp.wsloop reduction(@[[MIN_DECLARE_I]] -> %[[X_DECL]]#0 : !fir.ref<i32>) for
!CHECK:       fir.store %arg1 to %[[I_DECL]]#1 : !fir.ref<i32>
!CHECK:       %[[I_32:.*]] = fir.load %[[I_DECL]]#0 : !fir.ref<i32>
!CHECK:       %[[I_64:.*]] = fir.convert %[[I_32]] : (i32) -> i64
!CHECK:       %[[Y_I_REF:.*]] = hlfir.designate %[[Y_DECL]]#0 (%[[I_64]]) : (!fir.box<!fir.array<?xi32>>, i64) -> !fir.ref<i32>
!CHECK:       %[[Y_I_VAL:.*]] = fir.load %[[Y_I_REF]] : !fir.ref<i32>
!CHECK:       omp.reduction %[[Y_I_VAL]], %[[X_DECL]]#0 : i32, !fir.ref<i32>
!CHECK:       omp.yield
!CHECK:     omp.terminator

!CHECK-LABEL: @_QPreduction_min_real
!CHECK-SAME: %[[Y_BOX:.*]]: !fir.box<!fir.array<?xf32>>
!CHECK:   %[[X_REF:.*]] = fir.alloca f32 {bindc_name = "x", uniq_name = "_QFreduction_min_realEx"}
!CHECK:   %[[X_DECL:.*]]:2 = hlfir.declare %[[X_REF]] {uniq_name = "_QFreduction_min_realEx"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
!CHECK:   %[[Y_DECL:.*]]:2 = hlfir.declare %[[Y_BOX]] {uniq_name = "_QFreduction_min_realEy"} : (!fir.box<!fir.array<?xf32>>) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
!CHECK:   omp.parallel
!CHECK:      %[[I_REF:.*]] = fir.alloca i32 {adapt.valuebyref, pinned}
!CHECK:      %[[I_DECL:.*]]:2 = hlfir.declare %[[I_REF]] {uniq_name = "_QFreduction_min_realEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:     omp.wsloop reduction(@[[MIN_DECLARE_F]] -> %[[X_DECL]]#0 : !fir.ref<f32>) for
!CHECK:       fir.store %arg1 to %[[I_DECL]]#1 : !fir.ref<i32>
!CHECK:       %[[I_32:.*]] = fir.load %[[I_DECL]]#0 : !fir.ref<i32>
!CHECK:       %[[I_64:.*]] = fir.convert %[[I_32]] : (i32) -> i64
!CHECK:       %[[Y_I_REF:.*]] = hlfir.designate %[[Y_DECL]]#0 (%[[I_64]]) : (!fir.box<!fir.array<?xf32>>, i64) -> !fir.ref<f32>
!CHECK:       %[[Y_I_VAL:.*]] = fir.load %[[Y_I_REF]] : !fir.ref<f32>
!CHECK:       omp.yield
!CHECK:     omp.terminator

subroutine reduction_min_int(y)
  integer :: x, y(:)
  x = 0
  !$omp parallel
  !$omp do reduction(min:x)
  do i=1, 100
    x = min(x, y(i))
  end do
  !$omp end do
  !$omp end parallel
  print *, x
end subroutine

subroutine reduction_min_real(y)
  real :: x, y(:)
  x = 0.0
  !$omp parallel
  !$omp do reduction(min:x)
  do i=1, 100
    x = min(y(i), x)
  end do
  !$omp end do
  !$omp end parallel
  print *, x

  !$omp parallel
  !$omp do reduction(min:x)
  do i=1, 100
  !CHECK-NOT: omp.reduction
    if (y(i) .gt. x) x = y(i)
  end do
  !$omp end do
  !$omp end parallel
  print *, x
end subroutine

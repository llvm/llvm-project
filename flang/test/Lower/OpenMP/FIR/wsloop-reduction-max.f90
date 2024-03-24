! RUN: bbc -emit-fir -hlfir=false -fopenmp -o - %s 2>&1 | FileCheck %s
! RUN: %flang_fc1 -emit-fir -flang-deprecated-no-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s

!CHECK: omp.declare_reduction @[[MAX_DECLARE_F:.*]] : f32 init {
!CHECK:   %[[MINIMUM_VAL_F:.*]] = arith.constant -3.40282347E+38 : f32
!CHECK:   omp.yield(%[[MINIMUM_VAL_F]] : f32)
!CHECK: combiner
!CHECK: ^bb0(%[[ARG0_F:.*]]: f32, %[[ARG1_F:.*]]: f32):
!CHECK:   %[[COMB_VAL_F:.*]] = arith.maximumf %[[ARG0_F]], %[[ARG1_F]] {{.*}}: f32
!CHECK:   omp.yield(%[[COMB_VAL_F]] : f32)

!CHECK: omp.declare_reduction @[[MAX_DECLARE_I:.*]] : i32 init {
!CHECK:   %[[MINIMUM_VAL_I:.*]] = arith.constant -2147483648 : i32
!CHECK:   omp.yield(%[[MINIMUM_VAL_I]] : i32)
!CHECK: combiner
!CHECK: ^bb0(%[[ARG0_I:.*]]: i32, %[[ARG1_I:.*]]: i32):
!CHECK:   %[[COMB_VAL_I:.*]] = arith.maxsi %[[ARG0_I]], %[[ARG1_I]] : i32
!CHECK:   omp.yield(%[[COMB_VAL_I]] : i32)

!CHECK-LABEL: @_QPreduction_max_int
!CHECK-SAME: %[[Y_BOX:.*]]: !fir.box<!fir.array<?xi32>>
!CHECK:   %[[X_REF:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFreduction_max_intEx"}
!CHECK:   omp.parallel
!CHECK:     omp.wsloop reduction(@[[MAX_DECLARE_I]] %[[X_REF]] -> %[[PRV:.+]] : !fir.ref<i32>) for
!CHECK:       %[[LPRV:.+]] = fir.load %[[PRV]] : !fir.ref<i32>
!CHECK:       %[[Y_I_REF:.*]] = fir.coordinate_of %[[Y_BOX]]
!CHECK:       %[[Y_I:.*]] = fir.load %[[Y_I_REF]] : !fir.ref<i32>
!CHECK:       %[[RES:.+]] = arith.cmpi sgt, %[[LPRV]], %[[Y_I]] : i32
!CHECK:       %[[SEL:.+]] = arith.select %[[RES]], %[[LPRV]], %[[Y_I]]
!CHECK:       fir.store %[[SEL]] to %[[PRV]] : !fir.ref<i32>
!CHECK:     omp.terminator

!CHECK-LABEL: @_QPreduction_max_real
!CHECK-SAME: %[[Y_BOX:.*]]: !fir.box<!fir.array<?xf32>>
!CHECK:   %[[X_REF:.*]] = fir.alloca f32 {bindc_name = "x", uniq_name = "_QFreduction_max_realEx"}
!CHECK:   omp.parallel
!CHECK:     omp.wsloop reduction(@[[MAX_DECLARE_F]] %[[X_REF]] -> %[[PRV:.+]] : !fir.ref<f32>) for
!CHECK:       %[[LPRV:.+]] = fir.load %[[PRV]] : !fir.ref<f32>
!CHECK:       %[[Y_I_REF:.*]] = fir.coordinate_of %[[Y_BOX]]
!CHECK:       %[[Y_I:.*]] = fir.load %[[Y_I_REF]] : !fir.ref<f32>
!CHECK:       %[[RES:.+]] = arith.cmpf ogt, %[[Y_I]], %[[LPRV]] {{.*}} : f32
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

subroutine reduction_max_real(y)
  real :: x, y(:)
  x = 0.0
  !$omp parallel
  !$omp do reduction(max:x)
  do i=1, 100
    x = max(y(i), x)
  end do
  !$omp end do
  !$omp end parallel
  print *, x

  !$omp parallel
  !$omp do reduction(max:x)
  do i=1, 100
    !CHECK-NOT: omp.reduction
    if (y(i) .gt. x) x = y(i)
  end do
  !$omp end do
  !$omp end parallel
  print *, x
end subroutine

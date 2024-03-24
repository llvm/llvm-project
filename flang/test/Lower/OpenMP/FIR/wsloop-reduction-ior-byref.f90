! RUN: bbc -emit-fir -hlfir=false -fopenmp --force-byref-reduction %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir -flang-deprecated-no-hlfir -fopenmp -mmlir --force-byref-reduction %s -o - | FileCheck %s

! CHECK-LABEL:   omp.declare_reduction @ior_byref_i32 : !fir.ref<i32>
! CHECK-SAME:    init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<i32>):
! CHECK:            %[[C0_1:.*]] = arith.constant 0 : i32
! CHECK:            %[[REF:.*]] = fir.alloca i32
! CHECK:            fir.store %[[C0_1]] to %[[REF]] : !fir.ref<i32>
! CHECK:            omp.yield(%[[REF]] : !fir.ref<i32>)

! CHECK-LABEL:   } combiner {
! CHECK:         ^bb0(%[[ARG0:.*]]: !fir.ref<i32>, %[[ARG1:.*]]: !fir.ref<i32>):
! CHECK:           %[[LD0:.*]] = fir.load %[[ARG0]] : !fir.ref<i32>
! CHECK:           %[[LD1:.*]] = fir.load %[[ARG1]] : !fir.ref<i32>
! CHECK:           %[[RES:.*]] = arith.ori %[[LD0]], %[[LD1]] : i32
! CHECK:           fir.store %[[RES]] to %[[ARG0]] : !fir.ref<i32>
! CHECK:           omp.yield(%[[ARG0]] : !fir.ref<i32>)
! CHECK:         }

!CHECK-LABEL: @_QPreduction_ior
!CHECK-SAME: %[[Y_BOX:.*]]: !fir.box<!fir.array<?xi32>> 
!CHECK: %[[X_REF:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFreduction_iorEx"}
!CHECK: omp.parallel
!CHECK: omp.wsloop byref reduction(@ior_byref_i32 %[[X_REF]] -> %[[PRV:.+]] : !fir.ref<i32>) for 
!CHECK: %[[LPRV:.+]] = fir.load %[[PRV]] : !fir.ref<i32>
!CHECK: %[[Y_I_REF:.*]] = fir.coordinate_of %[[Y_BOX]]
!CHECK: %[[Y_I:.*]] = fir.load %[[Y_I_REF]] : !fir.ref<i32>
!CHECK: %[[RES:.+]] = arith.ori %[[LPRV]], %[[Y_I]] : i32
!CHECK: fir.store %[[RES]] to %[[PRV]] : !fir.ref<i32>
!CHECK: omp.yield
!CHECK: omp.terminator

subroutine reduction_ior(y)
  integer :: x, y(:)
  x = 0
  !$omp parallel
  !$omp do reduction(ior:x)
  do i=1, 100
    x = ior(x, y(i))
  end do
  !$omp end do
  !$omp end parallel
  print *, x
end subroutine

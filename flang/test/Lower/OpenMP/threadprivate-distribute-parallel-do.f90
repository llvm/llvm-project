! Check that threadprivate variables referenced in a composite DISTRIBUTE
! PARALLEL DO construct are bound inside the loop nest, since a composite
! omp.parallel may only contain its omp.distribute.

!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

!CHECK-LABEL: func.func @_QPdpd
!CHECK:         %[[CMN:.*]] = fir.address_of(@c_) : !fir.ref<!fir.array<4xi8>>
!CHECK:         omp.teams
!CHECK-NEXT:      omp.parallel
!CHECK-NOT:         omp.threadprivate
!CHECK:            omp.distribute
!CHECK-NEXT:          omp.wsloop
!CHECK-NEXT:            omp.loop_nest
!CHECK:                   omp.threadprivate %[[CMN]] : !fir.ref<!fir.array<4xi8>> -> !fir.ref<!fir.array<4xi8>>
subroutine dpd(n)
  integer :: n, i
  integer :: x
  common /c/ x
  !$omp threadprivate(/c/)

  !$omp teams distribute parallel do
  do i = 1, n
    x = i
  end do
end subroutine

!CHECK-LABEL: func.func @_QPdpd_simd
!CHECK:         %[[CMN:.*]] = fir.address_of(@c_) : !fir.ref<!fir.array<4xi8>>
!CHECK:         omp.teams
!CHECK-NEXT:      omp.parallel
!CHECK-NOT:         omp.threadprivate
!CHECK:            omp.distribute
!CHECK-NEXT:          omp.wsloop
!CHECK-NEXT:            omp.simd
!CHECK-NEXT:              omp.loop_nest
!CHECK:                     omp.threadprivate %[[CMN]] : !fir.ref<!fir.array<4xi8>> -> !fir.ref<!fir.array<4xi8>>
subroutine dpd_simd(n)
  integer :: n, i
  integer :: x
  common /c/ x
  !$omp threadprivate(/c/)

  !$omp teams distribute parallel do simd
  do i = 1, n
    x = i
  end do
end subroutine
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-is-target-device %s -o - | FileCheck %s

! Test user-defined declare reduction with intrinsic types on target device.
! These should generate inline constant initialization (no runtime calls),
! so they work on GPU targets without requiring the device Fortran runtime.

! CHECK-LABEL: omp.declare_reduction @_QQFaddc : complex<f32> init {
! CHECK:         %[[CZERO:.*]] = fir.zero_bits complex<f32>
! CHECK:         omp.yield(%[[CZERO]] : complex<f32>)
! CHECK:       } combiner {
! CHECK:         fir.addc
! CHECK:       }

! CHECK-LABEL: omp.declare_reduction @_QQFaddr : f32 init {
! CHECK:         %[[FZERO:.*]] = fir.zero_bits f32
! CHECK:         omp.yield(%[[FZERO]] : f32)
! CHECK:       } combiner {
! CHECK:         arith.addf
! CHECK:       }

! CHECK-LABEL: omp.declare_reduction @_QQFaddi : i32 init {
! CHECK:         %[[IZERO:.*]] = fir.zero_bits i32
! CHECK:         omp.yield(%[[IZERO]] : i32)
! CHECK:       } combiner {
! CHECK:         arith.addi
! CHECK:       }

! CHECK: omp.target
! CHECK:   omp.teams reduction(@_QQFaddi
! CHECK:     omp.wsloop reduction(@_QQFaddi

! CHECK: omp.target
! CHECK:   omp.teams reduction(@_QQFaddr
! CHECK:     omp.wsloop reduction(@_QQFaddr

! CHECK: omp.target
! CHECK:   omp.teams reduction(@_QQFaddc
! CHECK:     omp.wsloop reduction(@_QQFaddc

program test_target_named_reduction
  implicit none
  integer :: a, i
  real :: r
  complex :: c

  !$omp declare reduction(addi: integer: omp_out = omp_out + omp_in)
  !$omp declare reduction(addr: real: omp_out = omp_out + omp_in)
  !$omp declare reduction(addc: complex: omp_out = omp_out + omp_in)

  a = 0
  r = 0.0
  c = (0.0, 0.0)

  !$omp target teams distribute parallel do reduction(addi: a) map(tofrom: a)
  do i = 1, 10
    a = a + i
  end do
  !$omp end target teams distribute parallel do

  !$omp target teams distribute parallel do reduction(addr: r) map(tofrom: r)
  do i = 1, 10
    r = r + real(i)
  end do
  !$omp end target teams distribute parallel do

  !$omp target teams distribute parallel do reduction(addc: c) map(tofrom: c)
  do i = 1, 10
    c = c + cmplx(real(i), 0.0)
  end do
  !$omp end target teams distribute parallel do

  print *, "Int:", a, " Real:", r, " Complex:", c
end program

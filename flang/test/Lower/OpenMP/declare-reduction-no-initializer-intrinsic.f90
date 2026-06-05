! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

! Test declare reduction without initializer clause for intrinsic types.
! Without an initializer, the private variable should be zero-initialized.

! CHECK-DAG: omp.declare_reduction @char_max : !fir.ref<!fir.char<1,10>>
! CHECK:       init {
! CHECK:       %[[CHZERO:.*]] = fir.zero_bits !fir.char<1,10>
! CHECK:       fir.store %[[CHZERO]]
! CHECK:       } combiner {

! CHECK-DAG: omp.declare_reduction @add_reduction_z32 : complex<f32> init {
! CHECK:       %[[CZERO:.*]] = fir.zero_bits complex<f32>
! CHECK:       omp.yield(%[[CZERO]] : complex<f32>)
! CHECK:     } combiner {
! CHECK:       fir.addc
! CHECK:     }

! CHECK-DAG: omp.declare_reduction @add_reduction_f32 : f32 init {
! CHECK:       %[[FZERO:.*]] = fir.zero_bits f32
! CHECK:       omp.yield(%[[FZERO]] : f32)
! CHECK:     } combiner {
! CHECK:       arith.addf
! CHECK:     }

! CHECK-DAG: omp.declare_reduction @add_reduction_i32 : i32 init {
! CHECK:       %[[IZERO:.*]] = fir.zero_bits i32
! CHECK:       omp.yield(%[[IZERO]] : i32)
! CHECK:     } combiner {
! CHECK:       arith.addi
! CHECK:     }

program test_no_init_intrinsic
  implicit none
  integer :: a, i
  real :: r
  complex :: c
  character(len=10) :: s

  !$omp declare reduction(+: integer: omp_out = omp_out + omp_in)
  !$omp declare reduction(+: real: omp_out = omp_out + omp_in)
  !$omp declare reduction(+: complex: omp_out = omp_out + omp_in)
  !$omp declare reduction(char_max: character(len=10): omp_out = max(omp_out, omp_in))

  a = 0
  r = 0.0
  c = (0.0, 0.0)
  s = "aaa"

  ! Test integer reduction without initializer
  ! CHECK: omp.wsloop {{.*}} reduction(@add_reduction_i32
  !$omp parallel do reduction(+: a)
  do i = 1, 10
    a = a + i
  end do
  !$omp end parallel do

  ! Test real reduction without initializer
  ! CHECK: omp.wsloop {{.*}} reduction(@add_reduction_f32
  !$omp parallel do reduction(+: r)
  do i = 1, 10
    r = r + real(i)
  end do
  !$omp end parallel do

  ! Test complex reduction without initializer
  ! CHECK: omp.wsloop {{.*}} reduction(@add_reduction_z32
  !$omp parallel do reduction(+: c)
  do i = 1, 10
    c = c + cmplx(real(i), 0.0)
  end do
  !$omp end parallel do

  ! Test fixed-length character reduction without initializer
  ! CHECK: omp.wsloop {{.*}} reduction(byref @char_max
  !$omp parallel do reduction(char_max: s)
  do i = 1, 10
    continue
  end do
  !$omp end parallel do

  print *, a, r, c, s
end program

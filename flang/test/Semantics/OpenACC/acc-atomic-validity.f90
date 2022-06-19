! RUN: %python %S/../test_errors.py %s %flang -fopenacc

! Check OpenACC clause validity for the following construct and directive:
!   2.12 Atomic

program openacc_atomic_validity

  implicit none

  integer :: i
  integer, parameter :: N = 256
  integer, dimension(N) :: c

  !$acc parallel
  !$acc atomic update
  c(i) = c(i) + 1

  !$acc atomic update
  c(i) = c(i) + 1
  !$acc end atomic

  !$acc atomic write
  c(i) = 10

  !$acc atomic write
  c(i) = 10
  !$acc end atomic

  !$acc atomic read
  i = c(i)

  !$acc atomic read
  i = c(i)
  !$acc end atomic

  !$acc atomic capture
  c(i) = i
  i = i + 1
  !$acc end atomic
  !$acc end parallel

end program openacc_atomic_validity

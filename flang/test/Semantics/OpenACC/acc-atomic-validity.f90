! RUN: %python %S/../test_errors.py %s %flang -fopenacc

! Check OpenACC clause validity for the following construct and directive:
!   2.12 Atomic

program openacc_atomic_validity

  implicit none

  integer :: i
  integer, parameter :: N = 256
  integer, dimension(N) :: c
  logical :: l


  !$acc parallel
  !$acc atomic update
  c(i) = c(i) + 1

  !$acc atomic update
  c(i) = c(i) + 1
  !$acc end atomic

  !$acc atomic write
  c(i) = 10

  !$acc atomic write if(l)
  c(i) = 10

  !$acc atomic write
  c(i) = 10
  !$acc end atomic

  !$acc atomic write if(.true.)
  c(i) = 10
  !$acc end atomic

  !$acc atomic read
  i = c(i)
  
  !$acc atomic read if(.true.)
  i = c(i)

  !$acc atomic read
  i = c(i)
  !$acc end atomic

  !$acc atomic read if(l) 
  i = c(i)
  !$acc end atomic

  !ERROR: FINALIZE clause is not allowed on the ATOMIC READ FINALIZE IF(L)
  !$acc atomic read finalize if(l) 
  i = c(i)
  !$acc end atomic

  !$acc atomic capture
  c(i) = i
  i = i + 1
  !$acc end atomic

  !$acc atomic capture if(l .EQV. .false.)
  c(i) = i
  i = i + 1
  !$acc end atomic

  !$acc atomic update
  !ERROR: RHS of atomic update statement must be scalar
  !ERROR: LHS of atomic update statement must be scalar
  c = c + 1

  !$acc atomic update if(i == 0)
  c(i) = c(i) + 1

  !ERROR: At most one IF clause can appear on the ATOMIC UPDATE IF(I == 0) IF(.TRUE.)
  !$acc atomic update if(i == 0) if(.true.)
  c(i) = c(i) + 1

  !$acc end parallel

end program openacc_atomic_validity

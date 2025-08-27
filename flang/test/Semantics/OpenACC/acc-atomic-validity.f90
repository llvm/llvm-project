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

  !TODO: Should error because c(i) references i which is the atomic update variable.
  !$acc atomic capture
  c(i) = i
  i = i + 1
  !$acc end atomic

  !ERROR: The variables assigned in this atomic capture construct must be distinct
  !$acc atomic capture
  c(1) = c(2)
  c(1) = c(3)
  !$acc end atomic
  
  !ERROR: The assignments in this atomic capture construct do not update a variable and capture either its initial or final value
  !$acc atomic capture
  c(1) = c(2)
  c(2) = c(2)
  !$acc end atomic

  !ERROR: The assignments in this atomic capture construct do not update a variable and capture either its initial or final value
  !$acc atomic capture
  c(1) = c(2)
  c(2) = c(1)
  !$acc end atomic

  !ERROR: The assignments in this atomic capture construct do not update a variable and capture either its initial or final value
  !$acc atomic capture
  c(1) = c(2)
  c(3) = c(2)
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

subroutine capture_with_convert_f64_to_i32()
  integer :: x
  real(8) :: v, w
  x = 1
  v = 0
  w = 2

  !$acc atomic capture
  x = x * 2.5_8
  v = x
  !$acc end atomic

  !$acc atomic capture
  !TODO: The rhs side of this update statement cannot reference v.
  x = x * v
  v = x
  !$acc end atomic

  !$acc atomic capture
  !TODO: The rhs side of this update statement cannot reference v.
  x = v * x
  v = x
  !$acc end atomic

  !$acc atomic capture
  !ERROR: The RHS of this atomic update statement must reference the updated variable: x
  x = v * v
  v = x
  !$acc end atomic

  !$acc atomic capture
  x = v
  !ERROR: The updated variable, v, cannot appear more than once in the atomic update operation
  v = v * v
  !$acc end atomic

  !$acc atomic capture
  v = x
  x = w * w
  !$acc end atomic
end subroutine capture_with_convert_f64_to_i32
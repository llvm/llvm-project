! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp
! OpenMP Version 4.5
! 2.15.3.6 Reduction Clause
program omp_reduction
  integer :: i
  real :: r
  character :: c
  complex :: z
  logical :: l

  ! * is allowed for integer, real, and complex
  ! but not for logical or character
  ! ERROR: The type of 'c' is incompatible with the reduction operator.
  ! ERROR: The type of 'l' is incompatible with the reduction operator.
  !$omp parallel reduction(*:i,r,c,z,l)
  !$omp end parallel

  ! + is allowed for integer, real, and complex
  ! but not for logical or character
  ! ERROR: The type of 'c' is incompatible with the reduction operator.
  ! ERROR: The type of 'l' is incompatible with the reduction operator.
  !$omp parallel reduction(+:i,r,c,z,l)
  !$omp end parallel

  ! - is deprecated for all types
  ! ERROR: The minus reduction operator is deprecated since OpenMP 5.2 and is not supported in the REDUCTION clause.
  !$omp parallel reduction(-:i,r,c,z,l)
  !$omp end parallel

  ! .and. is only supported for logical operations
  ! ERROR: The type of 'i' is incompatible with the reduction operator.
  ! ERROR: The type of 'r' is incompatible with the reduction operator.
  ! ERROR: The type of 'c' is incompatible with the reduction operator.
  ! ERROR: The type of 'z' is incompatible with the reduction operator.
  !$omp parallel reduction(.and.:i,r,c,z,l)
  !$omp end parallel

  ! .or. is only supported for logical operations
  ! ERROR: The type of 'i' is incompatible with the reduction operator.
  ! ERROR: The type of 'r' is incompatible with the reduction operator.
  ! ERROR: The type of 'c' is incompatible with the reduction operator.
  ! ERROR: The type of 'z' is incompatible with the reduction operator.
  !$omp parallel reduction(.or.:i,r,c,z,l)
  !$omp end parallel

  ! .eqv. is only supported for logical operations
  ! ERROR: The type of 'i' is incompatible with the reduction operator.
  ! ERROR: The type of 'r' is incompatible with the reduction operator.
  ! ERROR: The type of 'c' is incompatible with the reduction operator.
  ! ERROR: The type of 'z' is incompatible with the reduction operator.
  !$omp parallel reduction(.eqv.:i,r,c,z,l)
  !$omp end parallel

  ! .neqv. is only supported for logical operations
  ! ERROR: The type of 'i' is incompatible with the reduction operator.
  ! ERROR: The type of 'r' is incompatible with the reduction operator.
  ! ERROR: The type of 'c' is incompatible with the reduction operator.
  ! ERROR: The type of 'z' is incompatible with the reduction operator.
  !$omp parallel reduction(.neqv.:i,r,c,z,l)
  !$omp end parallel

  ! iand only supports integers
  ! ERROR: The type of 'r' is incompatible with the reduction operator.
  ! ERROR: The type of 'c' is incompatible with the reduction operator.
  ! ERROR: The type of 'z' is incompatible with the reduction operator.
  ! ERROR: The type of 'l' is incompatible with the reduction operator.
  !$omp parallel reduction(iand:i,r,c,z,l)
  !$omp end parallel

  ! ior only supports integers
  ! ERROR: The type of 'r' is incompatible with the reduction operator.
  ! ERROR: The type of 'c' is incompatible with the reduction operator.
  ! ERROR: The type of 'z' is incompatible with the reduction operator.
  ! ERROR: The type of 'l' is incompatible with the reduction operator.
  !$omp parallel reduction(ior:i,r,c,z,l)
  !$omp end parallel

  ! ieor only supports integers
  ! ERROR: The type of 'r' is incompatible with the reduction operator.
  ! ERROR: The type of 'c' is incompatible with the reduction operator.
  ! ERROR: The type of 'z' is incompatible with the reduction operator.
  ! ERROR: The type of 'l' is incompatible with the reduction operator.
  !$omp parallel reduction(ieor:i,r,c,z,l)
  !$omp end parallel

  ! max arguments may be integer, real, or character:
  ! ERROR: The type of 'z' is incompatible with the reduction operator.
  ! ERROR: The type of 'l' is incompatible with the reduction operator.
  !$omp parallel reduction(max:i,r,c,z,l)
  !$omp end parallel

  ! min arguments may be integer, real, or character:
  ! ERROR: The type of 'z' is incompatible with the reduction operator.
  ! ERROR: The type of 'l' is incompatible with the reduction operator.
  !$omp parallel reduction(min:i,r,c,z,l)
  !$omp end parallel
end program omp_reduction

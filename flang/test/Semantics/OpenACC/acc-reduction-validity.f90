! RUN: %python %S/../test_errors.py %s %flang -fopenacc

! Check OpenACC reduction validity.

program openacc_reduction_validity
  implicit none

  integer :: i
  real :: r
  complex :: c
  logical :: l

  !$acc parallel reduction(+:i)
  !$acc end parallel

  !$acc parallel reduction(*:i)
  !$acc end parallel

  !$acc parallel reduction(min:i)
  !$acc end parallel

  !$acc parallel reduction(max:i)
  !$acc end parallel

  !$acc parallel reduction(iand:i)
  !$acc end parallel

  !$acc parallel reduction(ior:i)
  !$acc end parallel

  !$acc parallel reduction(ieor:i)
  !$acc end parallel

  !ERROR: reduction operator not supported for integer type
  !$acc parallel reduction(.and.:i)
  !$acc end parallel

  !ERROR: reduction operator not supported for integer type
  !$acc parallel reduction(.or.:i)
  !$acc end parallel

  !ERROR: reduction operator not supported for integer type
  !$acc parallel reduction(.eqv.:i)
  !$acc end parallel

  !ERROR: reduction operator not supported for integer type
  !$acc parallel reduction(.neqv.:i)
  !$acc end parallel

  !$acc parallel reduction(+:r)
  !$acc end parallel

  !$acc parallel reduction(*:r)
  !$acc end parallel

  !$acc parallel reduction(min:r)
  !$acc end parallel

  !$acc parallel reduction(max:r)
  !$acc end parallel

  !ERROR: reduction operator not supported for real type
  !$acc parallel reduction(iand:r)
  !$acc end parallel

  !ERROR: reduction operator not supported for real type
  !$acc parallel reduction(ior:r)
  !$acc end parallel

  !ERROR: reduction operator not supported for real type
  !$acc parallel reduction(ieor:r)
  !$acc end parallel

  !ERROR: reduction operator not supported for real type
  !$acc parallel reduction(.and.:r)
  !$acc end parallel

  !ERROR: reduction operator not supported for real type
  !$acc parallel reduction(.or.:r)
  !$acc end parallel

  !ERROR: reduction operator not supported for real type
  !$acc parallel reduction(.eqv.:r)
  !$acc end parallel

  !ERROR: reduction operator not supported for real type
  !$acc parallel reduction(.neqv.:r)
  !$acc end parallel

  !$acc parallel reduction(+:c)
  !$acc end parallel

  !$acc parallel reduction(*:c)
  !$acc end parallel

  !ERROR: reduction operator not supported for complex type
  !$acc parallel reduction(min:c)
  !$acc end parallel

  !ERROR: reduction operator not supported for complex type
  !$acc parallel reduction(max:c)
  !$acc end parallel

  !ERROR: reduction operator not supported for complex type
  !$acc parallel reduction(iand:c)
  !$acc end parallel

  !ERROR: reduction operator not supported for complex type
  !$acc parallel reduction(ior:c)
  !$acc end parallel

  !ERROR: reduction operator not supported for complex type
  !$acc parallel reduction(ieor:c)
  !$acc end parallel

  !ERROR: reduction operator not supported for complex type
  !$acc parallel reduction(.and.:c)
  !$acc end parallel

  !ERROR: reduction operator not supported for complex type
  !$acc parallel reduction(.or.:c)
  !$acc end parallel

  !ERROR: reduction operator not supported for complex type
  !$acc parallel reduction(.eqv.:c)
  !$acc end parallel

  !ERROR: reduction operator not supported for complex type
  !$acc parallel reduction(.neqv.:c)
  !$acc end parallel

  !$acc parallel reduction(.and.:l)
  !$acc end parallel

  !$acc parallel reduction(.or.:l)
  !$acc end parallel

  !$acc parallel reduction(.eqv.:l)
  !$acc end parallel

  !$acc parallel reduction(.neqv.:l)
  !$acc end parallel

  !ERROR: reduction operator not supported for logical type
  !$acc parallel reduction(+:l)
  !$acc end parallel

  !ERROR: reduction operator not supported for logical type
  !$acc parallel reduction(*:l)
  !$acc end parallel

  !ERROR: reduction operator not supported for logical type
  !$acc parallel reduction(min:l)
  !$acc end parallel

  !ERROR: reduction operator not supported for logical type
  !$acc parallel reduction(max:l)
  !$acc end parallel

  !ERROR: reduction operator not supported for logical type
  !$acc parallel reduction(iand:l)
  !$acc end parallel

  !ERROR: reduction operator not supported for logical type
  !$acc parallel reduction(ior:l)
  !$acc end parallel

  !ERROR: reduction operator not supported for logical type
  !$acc parallel reduction(ieor:l)
  !$acc end parallel

  !ERROR: No explicit type declared for 'xyz'
  !$acc parallel reduction(+:xyz)
  !$acc end parallel  


end program

subroutine sum()
  !ERROR: 'sum' is already declared in this scoping unit
  integer :: i,sum 
  sum = 0
  !$acc parallel
  !ERROR: Only variables are allowed in data clauses on the LOOP directive
  !$acc loop independent gang reduction(+:sum)
  do i=1,10
     sum = sum + i
  enddo
  !$acc end parallel
end subroutine

subroutine reduce()
  integer :: red = 0, ii
  !$acc parallel loop default(none) reduction(+:red)
  do ii = 1, 10
    red = red + ii
  end do
  !$acc end parallel
end subroutine

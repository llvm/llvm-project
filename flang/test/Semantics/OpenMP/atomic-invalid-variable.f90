! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp
! Test for issue #169484: Flang should not crash on invalid atomic variables
! This test verifies that proper diagnostics are emitted for invalid atomic
! constructs instead of crashing the compiler.

subroutine test_atomic_invalid(a, b, i)
  integer :: i, a, b
  interface
    function baz(i) result(res)
      integer :: i, res
    end function
  end interface
  
  ! Valid atomic update - should work fine
  !$omp atomic
    b = b + a
  
  ! Invalid: z is undeclared, so z(1) is treated as a function reference
  ! This should emit an error, not crash
  !$omp atomic
    !ERROR: Left-hand side of assignment is not definable
    !ERROR: 'z(1_4)' is not a variable or pointer
    z(1) = z(1) + 1
  
  ! Invalid: baz(i) is a function reference, not a variable
  ! This should emit an error, not crash
  !$omp atomic
    !ERROR: Left-hand side of assignment is not definable
    !ERROR: 'baz(i)' is not a variable or pointer
    !ERROR: This is not a valid ATOMIC UPDATE operation
    baz(i) = 1
end subroutine

! REQUIRES: openmp_runtime

! RUN: %python %S/../test_errors.py %s %flang_fc1 %openmp_flags

! OpenMP Atomic construct
! section 2.17.7
! Intrinsic procedure name is one of MAX, MIN, IAND, IOR, or IEOR.

program OmpAtomic
   use omp_lib
   real x
   integer :: y, z, a, b, c, d
   x = 5.73
   y = 3
   z = 1
!$omp atomic
   y = IAND(y, 4)
!$omp atomic
   y = IOR(y, 5)
!$omp atomic
   y = IEOR(y, 6)
!$omp atomic
   y = MAX(y, 7)
!$omp atomic
   y = MIN(y, 8)

!$omp atomic
   !ERROR: The atomic variable z should occur exactly once among the arguments of the top-level AND operator
   z = IAND(y, 4)
!$omp atomic
   !ERROR: The atomic variable z should occur exactly once among the arguments of the top-level OR operator
   z = IOR(y, 5)
!$omp atomic
   !ERROR: The atomic variable z should occur exactly once among the arguments of the top-level NEQV/EOR operator
   z = IEOR(y, 6)
!$omp atomic
   !ERROR: The atomic variable z should occur exactly once among the arguments of the top-level MAX operator
   z = MAX(y, 7, b, c)
!$omp atomic
   !ERROR: The atomic variable z should occur exactly once among the arguments of the top-level MIN operator
   z = MIN(y, 8, a, d)

!$omp atomic
   !ERROR: This intrinsic function is not a valid ATOMIC UPDATE operation
   y = FRACTION(x)
!$omp atomic
   !ERROR: The atomic variable y should appear as an argument in the update operation
   y = REAL(x)
!$omp atomic update
   y = IAND(y, 4)
!$omp atomic update
   y = IOR(y, 5)
!$omp atomic update
   y = IEOR(y, 6)
!$omp atomic update
   y = MAX(y, 7)
!$omp atomic update
   y = MIN(y, 8)

!$omp atomic update
   !ERROR: The atomic variable z should occur exactly once among the arguments of the top-level AND operator
   z = IAND(y, 4)
!$omp atomic update 
   !ERROR: The atomic variable z should occur exactly once among the arguments of the top-level OR operator
   z = IOR(y, 5)
!$omp atomic update
   !ERROR: The atomic variable z should occur exactly once among the arguments of the top-level NEQV/EOR operator
   z = IEOR(y, 6)
!$omp atomic update
   !ERROR: The atomic variable z should occur exactly once among the arguments of the top-level MAX operator
   z = MAX(y, 7)
!$omp atomic update
   !ERROR: The atomic variable z should occur exactly once among the arguments of the top-level MIN operator
   z = MIN(y, 8)

!$omp atomic update
  !ERROR: This intrinsic function is not a valid ATOMIC UPDATE operation
   y = MOD(y, 9)
!$omp atomic update
  !ERROR: This intrinsic function is not a valid ATOMIC UPDATE operation
   x = ABS(x)
end program OmpAtomic

subroutine conflicting_types()
    type simple
    integer :: z
    end type
    real x
    integer :: y, z
    type(simple) ::s
    z = 1
    !$omp atomic
    !ERROR: The atomic variable z should occur exactly once among the arguments of the top-level AND operator
    z = IAND(s%z, 4)
end subroutine

subroutine more_invalid_atomic_update_stmts()
    integer :: a, b
    integer :: k(10)
    type some_type
        integer :: m(10)
    end type
    type(some_type) :: s
 
    !$omp atomic update
    !ERROR: The atomic variable a should occur exactly once among the arguments of the top-level MIN operator
        a = min(a, a, b)
     
    !$omp atomic
    !ERROR: The atomic variable a should occur exactly once among the arguments of the top-level MAX operator
        a = max(b, a, b, a)

    !$omp atomic
        a = min(b, a, b)

    !$omp atomic
    !ERROR: The atomic variable a should occur exactly once among the arguments of the top-level MAX operator
        a = max(b, a, b, a, b)
    
    !$omp atomic update
    !ERROR: The atomic variable y should occur exactly once among the arguments of the top-level MIN operator
        y = min(z, x)
     
    !$omp atomic
        z = max(z, y)

    !$omp atomic update
    !ERROR: Atomic variable k should be a scalar
    !ERROR: The atomic variable k should occur exactly once among the arguments of the top-level MAX operator
        k = max(x, y)

    !$omp atomic
    !ERROR: No intrinsic or user-defined ASSIGNMENT(=) matches scalar REAL(4) and rank 1 array of REAL(4)
        x = min(x, k)

    !$omp atomic
    !ERROR: No intrinsic or user-defined ASSIGNMENT(=) matches scalar REAL(4) and rank 1 array of REAL(4)
        z = z + s%m
end subroutine

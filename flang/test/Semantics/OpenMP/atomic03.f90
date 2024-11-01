! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp

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
   !ERROR: Atomic update variable 'z' not found in the argument list of intrinsic procedure
   z = IAND(y, 4)
!$omp atomic
   !ERROR: Atomic update variable 'z' not found in the argument list of intrinsic procedure
   z = IOR(y, 5)
!$omp atomic
   !ERROR: Atomic update variable 'z' not found in the argument list of intrinsic procedure
   z = IEOR(y, 6)
!$omp atomic
   !ERROR: Atomic update variable 'z' not found in the argument list of intrinsic procedure
   z = MAX(y, 7, b, c)
!$omp atomic
   !ERROR: Atomic update variable 'z' not found in the argument list of intrinsic procedure
   z = MIN(y, 8, a, d)

!$omp atomic
   !ERROR: Invalid intrinsic procedure name in OpenMP ATOMIC (UPDATE) statement
   y = FRACTION(x)
!$omp atomic
   !ERROR: Invalid intrinsic procedure name in OpenMP ATOMIC (UPDATE) statement
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
   !ERROR: Atomic update variable 'z' not found in the argument list of intrinsic procedure
   z = IAND(y, 4)
!$omp atomic update
   !ERROR: Atomic update variable 'z' not found in the argument list of intrinsic procedure
   z = IOR(y, 5)
!$omp atomic update
   !ERROR: Atomic update variable 'z' not found in the argument list of intrinsic procedure
   z = IEOR(y, 6)
!$omp atomic update
   !ERROR: Atomic update variable 'z' not found in the argument list of intrinsic procedure
   z = MAX(y, 7)
!$omp atomic update
   !ERROR: Atomic update variable 'z' not found in the argument list of intrinsic procedure
   z = MIN(y, 8)

!$omp atomic update
   !ERROR: Invalid intrinsic procedure name in OpenMP ATOMIC (UPDATE) statement
   y = MOD(y, 9)
!$omp atomic update
   !ERROR: Invalid intrinsic procedure name in OpenMP ATOMIC (UPDATE) statement
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
    !ERROR: Atomic update variable 'z' not found in the argument list of intrinsic procedure
    z = IAND(s%z, 4)
end subroutine

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
   !ERROR: Intrinsic procedure arguments in atomic update statement must have exactly one occurence of 'z'
   z = IAND(y, 4)
!$omp atomic
   !ERROR: Intrinsic procedure arguments in atomic update statement must have exactly one occurence of 'z'
   z = IOR(y, 5)
!$omp atomic
   !ERROR: Intrinsic procedure arguments in atomic update statement must have exactly one occurence of 'z'
   z = IEOR(y, 6)
!$omp atomic
   !ERROR: Intrinsic procedure arguments in atomic update statement must have exactly one occurence of 'z'
   z = MAX(y, 7, b, c)
!$omp atomic
   !ERROR: Intrinsic procedure arguments in atomic update statement must have exactly one occurence of 'z'
   z = MIN(y, 8, a, d)

!$omp atomic
   !ERROR: Invalid intrinsic procedure name in OpenMP ATOMIC (UPDATE) statement
   !ERROR: Intrinsic procedure arguments in atomic update statement must have exactly one occurence of 'y'
   y = FRACTION(x)
!$omp atomic
   !ERROR: Invalid intrinsic procedure name in OpenMP ATOMIC (UPDATE) statement
   !ERROR: Intrinsic procedure arguments in atomic update statement must have exactly one occurence of 'y'
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
   !ERROR: Intrinsic procedure arguments in atomic update statement must have exactly one occurence of 'z'
   z = IAND(y, 4)
!$omp atomic update 
   !ERROR: Intrinsic procedure arguments in atomic update statement must have exactly one occurence of 'z'
   z = IOR(y, 5)
!$omp atomic update
   !ERROR: Intrinsic procedure arguments in atomic update statement must have exactly one occurence of 'z'
   z = IEOR(y, 6)
!$omp atomic update
   !ERROR: Intrinsic procedure arguments in atomic update statement must have exactly one occurence of 'z'
   z = MAX(y, 7)
!$omp atomic update
   !ERROR: Intrinsic procedure arguments in atomic update statement must have exactly one occurence of 'z'
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
    !ERROR: Intrinsic procedure arguments in atomic update statement must have exactly one occurence of 'z'
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
    !ERROR: Intrinsic procedure arguments in atomic update statement must have exactly one occurence of 'a'
        a = min(a, a, b)
     
    !$omp atomic
    !ERROR: Intrinsic procedure arguments in atomic update statement must have exactly one occurence of 'a'
        a = max(b, a, b, a)

    !$omp atomic
    !ERROR: Atomic update statement should be of the form `a = intrinsic_procedure(a, expr_list)` OR `a = intrinsic_procedure(expr_list, a)`
        a = min(b, a, b)

    !$omp atomic
    !ERROR: Intrinsic procedure arguments in atomic update statement must have exactly one occurence of 'a'
        a = max(b, a, b, a, b)
    
    !$omp atomic update
    !ERROR: Intrinsic procedure arguments in atomic update statement must have exactly one occurence of 'y'
        y = min(z, x)
     
    !$omp atomic
        z = max(z, y)

    !$omp atomic update
    !ERROR: Expected scalar variable on the LHS of atomic update assignment statement
    !ERROR: Intrinsic procedure arguments in atomic update statement must have exactly one occurence of 'k'
        k = max(x, y)
    
    !$omp atomic
    !ERROR: No intrinsic or user-defined ASSIGNMENT(=) matches scalar REAL(4) and rank 1 array of REAL(4)
    !ERROR: Expected scalar expression on the RHS of atomic update assignment statement
        x = min(x, k)

    !$omp atomic
    !ERROR: No intrinsic or user-defined ASSIGNMENT(=) matches scalar REAL(4) and rank 1 array of REAL(4)
    !ERROR: Expected scalar expression on the RHS of atomic update assignment statement
        z =z + s%m
end subroutine

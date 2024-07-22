! REQUIRES: openmp_runtime

! RUN: %python %S/../test_errors.py %s %flang_fc1 %openmp_flags

! OpenMP Atomic construct
! section 2.17.7
! Update assignment must be 'var = var op expr' or 'var = expr op var'

program OmpAtomic
   use omp_lib
   real x
   integer y
   logical m, n, l
   x = 5.73
   y = 3
   m = .TRUE.
   n = .FALSE.
!$omp atomic
   x = x + 1
!$omp atomic
   x = 1 + x
!$omp atomic
   !ERROR: Atomic update statement should be of form `x = x operator expr` OR `x = expr operator x`
   !ERROR: Exactly one occurence of 'x' expected on the RHS of atomic update assignment statement
   x = y + 1
!$omp atomic
   !ERROR: Atomic update statement should be of form `x = x operator expr` OR `x = expr operator x`
   !ERROR: Exactly one occurence of 'x' expected on the RHS of atomic update assignment statement
   x = 1 + y

!$omp atomic
   x = x - 1
!$omp atomic
   x = 1 - x
!$omp atomic
   !ERROR: Atomic update statement should be of form `x = x operator expr` OR `x = expr operator x`
   !ERROR: Exactly one occurence of 'x' expected on the RHS of atomic update assignment statement
   x = y - 1
!$omp atomic
   !ERROR: Atomic update statement should be of form `x = x operator expr` OR `x = expr operator x`
   !ERROR: Exactly one occurence of 'x' expected on the RHS of atomic update assignment statement
   x = 1 - y

!$omp atomic
   x = x*1
!$omp atomic
   x = 1*x
!$omp atomic
   !ERROR: Atomic update statement should be of form `x = x operator expr` OR `x = expr operator x`
   !ERROR: Exactly one occurence of 'x' expected on the RHS of atomic update assignment statement
   x = y*1
!$omp atomic
   !ERROR: Atomic update statement should be of form `x = x operator expr` OR `x = expr operator x`
   !ERROR: Exactly one occurence of 'x' expected on the RHS of atomic update assignment statement
   x = 1*y

!$omp atomic
   x = x/1
!$omp atomic
   x = 1/x
!$omp atomic
   !ERROR: Atomic update statement should be of form `x = x operator expr` OR `x = expr operator x`
   !ERROR: Exactly one occurence of 'x' expected on the RHS of atomic update assignment statement
   x = y/1
!$omp atomic
   !ERROR: Atomic update statement should be of form `x = x operator expr` OR `x = expr operator x`
   !ERROR: Exactly one occurence of 'x' expected on the RHS of atomic update assignment statement
   x = 1/y

!$omp atomic
   m = m .AND. n
!$omp atomic
   m = n .AND. m
!$omp atomic 
   !ERROR: Atomic update statement should be of form `m = m operator expr` OR `m = expr operator m`
   !ERROR: Exactly one occurence of 'm' expected on the RHS of atomic update assignment statement
   m = n .AND. l

!$omp atomic
   m = m .OR. n
!$omp atomic
   m = n .OR. m
!$omp atomic 
   !ERROR: Atomic update statement should be of form `m = m operator expr` OR `m = expr operator m`
   !ERROR: Exactly one occurence of 'm' expected on the RHS of atomic update assignment statement
   m = n .OR. l

!$omp atomic
   m = m .EQV. n
!$omp atomic
   m = n .EQV. m
!$omp atomic
   !ERROR: Atomic update statement should be of form `m = m operator expr` OR `m = expr operator m`
   !ERROR: Exactly one occurence of 'm' expected on the RHS of atomic update assignment statement
   m = n .EQV. l

!$omp atomic
   m = m .NEQV. n
!$omp atomic
   m = n .NEQV. m
!$omp atomic
   !ERROR: Atomic update statement should be of form `m = m operator expr` OR `m = expr operator m`
   !ERROR: Exactly one occurence of 'm' expected on the RHS of atomic update assignment statement
   m = n .NEQV. l

!$omp atomic update
   x = x + 1
!$omp atomic update
   x = 1 + x
!$omp atomic update
   !ERROR: Atomic update statement should be of form `x = x operator expr` OR `x = expr operator x`
   !ERROR: Exactly one occurence of 'x' expected on the RHS of atomic update assignment statement
   x = y + 1
!$omp atomic update
   !ERROR: Atomic update statement should be of form `x = x operator expr` OR `x = expr operator x`
   !ERROR: Exactly one occurence of 'x' expected on the RHS of atomic update assignment statement
   x = 1 + y

!$omp atomic update
   x = x - 1
!$omp atomic update
   x = 1 - x
!$omp atomic update
   !ERROR: Atomic update statement should be of form `x = x operator expr` OR `x = expr operator x`
   !ERROR: Exactly one occurence of 'x' expected on the RHS of atomic update assignment statement
   x = y - 1
!$omp atomic update
   !ERROR: Atomic update statement should be of form `x = x operator expr` OR `x = expr operator x`
   !ERROR: Exactly one occurence of 'x' expected on the RHS of atomic update assignment statement
   x = 1 - y

!$omp atomic update
   x = x*1
!$omp atomic update
   x = 1*x
!$omp atomic update
   !ERROR: Atomic update statement should be of form `x = x operator expr` OR `x = expr operator x`
   !ERROR: Exactly one occurence of 'x' expected on the RHS of atomic update assignment statement
   x = y*1
!$omp atomic update
   !ERROR: Atomic update statement should be of form `x = x operator expr` OR `x = expr operator x`
   !ERROR: Exactly one occurence of 'x' expected on the RHS of atomic update assignment statement
   x = 1*y

!$omp atomic update
   x = x/1
!$omp atomic update
   x = 1/x
!$omp atomic update
   !ERROR: Atomic update statement should be of form `x = x operator expr` OR `x = expr operator x`
   !ERROR: Exactly one occurence of 'x' expected on the RHS of atomic update assignment statement
   x = y/1
!$omp atomic update
   !ERROR: Atomic update statement should be of form `x = x operator expr` OR `x = expr operator x`
   !ERROR: Exactly one occurence of 'x' expected on the RHS of atomic update assignment statement
   x = 1/y

!$omp atomic update
   m = m .AND. n
!$omp atomic update
   m = n .AND. m
!$omp atomic update
   !ERROR: Atomic update statement should be of form `m = m operator expr` OR `m = expr operator m`
   !ERROR: Exactly one occurence of 'm' expected on the RHS of atomic update assignment statement
   m = n .AND. l

!$omp atomic update
   m = m .OR. n
!$omp atomic update
   m = n .OR. m
!$omp atomic update
   !ERROR: Atomic update statement should be of form `m = m operator expr` OR `m = expr operator m`
   !ERROR: Exactly one occurence of 'm' expected on the RHS of atomic update assignment statement
   m = n .OR. l

!$omp atomic update
   m = m .EQV. n
!$omp atomic update
   m = n .EQV. m
!$omp atomic update
   !ERROR: Atomic update statement should be of form `m = m operator expr` OR `m = expr operator m`
   !ERROR: Exactly one occurence of 'm' expected on the RHS of atomic update assignment statement
   m = n .EQV. l

!$omp atomic update
   m = m .NEQV. n
!$omp atomic update
   m = n .NEQV. m
!$omp atomic update
   !ERROR: Atomic update statement should be of form `m = m operator expr` OR `m = expr operator m`
   !ERROR: Exactly one occurence of 'm' expected on the RHS of atomic update assignment statement
   m = n .NEQV. l

end program OmpAtomic

subroutine more_invalid_atomic_update_stmts()
    integer :: a, b, c
    integer :: d(10)
    real :: x, y, z(10)
    type some_type
        real :: m
        real :: n(10)
    end type
    type(some_type) p
    
    !$omp atomic
    !ERROR: Invalid or missing operator in atomic update statement
        x = x

    !$omp atomic update
    !ERROR: Invalid or missing operator in atomic update statement
        x = 1    

    !$omp atomic update
    !ERROR: Exactly one occurence of 'a' expected on the RHS of atomic update assignment statement
        a = a * b + a

    !$omp atomic
    !ERROR: Atomic update statement should be of form `a = a operator expr` OR `a = expr operator a`
        a = b * (a + 9)

    !$omp atomic update
    !ERROR: Exactly one occurence of 'a' expected on the RHS of atomic update assignment statement
        a = a * (a + b)

    !$omp atomic
    !ERROR: Exactly one occurence of 'a' expected on the RHS of atomic update assignment statement
        a = (b + a) * a

    !$omp atomic
    !ERROR: Atomic update statement should be of form `a = a operator expr` OR `a = expr operator a`
        a = a * b + c

    !$omp atomic update
    !ERROR: Atomic update statement should be of form `a = a operator expr` OR `a = expr operator a`
        a = a + b + c

    !$omp atomic
        a = b * c + a

    !$omp atomic update
        a = c + b + a

    !$omp atomic
    !ERROR: No intrinsic or user-defined ASSIGNMENT(=) matches scalar INTEGER(4) and rank 1 array of INTEGER(4)
    !ERROR: Expected scalar expression on the RHS of atomic update assignment statement
        a = a + d

    !$omp atomic update
    !ERROR: No intrinsic or user-defined ASSIGNMENT(=) matches scalar REAL(4) and rank 1 array of REAL(4)
    !ERROR: Atomic update statement should be of form `x = x operator expr` OR `x = expr operator x`
    !ERROR: Expected scalar expression on the RHS of atomic update assignment statement
        x = x * y / z

    !$omp atomic
    !ERROR: Atomic update statement should be of form `p%m = p%m operator expr` OR `p%m = expr operator p%m`
    !ERROR: Exactly one occurence of 'p%m' expected on the RHS of atomic update assignment statement
        p%m = x + y

    !$omp atomic update
    !ERROR: No intrinsic or user-defined ASSIGNMENT(=) matches scalar REAL(4) and rank 1 array of REAL(4)
    !ERROR: Expected scalar expression on the RHS of atomic update assignment statement
    !ERROR: Exactly one occurence of 'p%m' expected on the RHS of atomic update assignment statement
        p%m = p%m + p%n
end subroutine

! RUN: %python %S/../test_errors.py %s %flang_fc1 %openmp_flags

! OpenMP Atomic construct
! section 2.17.7
! Update assignment must be 'var = var op expr' or 'var = expr op var'

program OmpAtomic
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
   !ERROR: The atomic variable x should appear as an argument of the top-level + operator
   x = y + 1
!$omp atomic
   !ERROR: The atomic variable x should appear as an argument of the top-level + operator
   x = 1 + y

!$omp atomic
   x = x - 1
!$omp atomic
   x = 1 - x
!$omp atomic
   !ERROR: The atomic variable x should appear as an argument of the top-level - operator
   x = y - 1
!$omp atomic
   !ERROR: The atomic variable x should appear as an argument of the top-level - operator
   x = 1 - y

!$omp atomic
   x = x*1
!$omp atomic
   x = 1*x
!$omp atomic
   !ERROR: The atomic variable x should appear as an argument in the update operation
   x = y*1
!$omp atomic
   !ERROR: The atomic variable x should appear as an argument in the update operation
   x = 1*y

!$omp atomic
   x = x/1
!$omp atomic
   x = 1/x
!$omp atomic
   !ERROR: The atomic variable x should appear as an argument of the top-level / operator
   x = y/1
!$omp atomic
   !ERROR: The atomic variable x should appear as an argument of the top-level / operator
   x = 1/y

!$omp atomic
   m = m .AND. n
!$omp atomic
   m = n .AND. m
!$omp atomic 
   !ERROR: The atomic variable m should appear as an argument of the top-level AND operator
   m = n .AND. l

!$omp atomic
   m = m .OR. n
!$omp atomic
   m = n .OR. m
!$omp atomic 
   !ERROR: The atomic variable m should appear as an argument of the top-level OR operator
   m = n .OR. l

!$omp atomic
   m = m .EQV. n
!$omp atomic
   m = n .EQV. m
!$omp atomic
   !ERROR: The atomic variable m should appear as an argument of the top-level EQV operator
   m = n .EQV. l

!$omp atomic
   m = m .NEQV. n
!$omp atomic
   m = n .NEQV. m
!$omp atomic
   !ERROR: The atomic variable m should appear as an argument of the top-level NEQV/EOR operator
   m = n .NEQV. l

!$omp atomic update
   x = x + 1
!$omp atomic update
   x = 1 + x
!$omp atomic update
   !ERROR: The atomic variable x should appear as an argument of the top-level + operator
   x = y + 1
!$omp atomic update
   !ERROR: The atomic variable x should appear as an argument of the top-level + operator
   x = 1 + y

!$omp atomic update
   x = x - 1
!$omp atomic update
   x = 1 - x
!$omp atomic update
   !ERROR: The atomic variable x should appear as an argument of the top-level - operator
   x = y - 1
!$omp atomic update
   !ERROR: The atomic variable x should appear as an argument of the top-level - operator
   x = 1 - y

!$omp atomic update
   x = x*1
!$omp atomic update
   x = 1*x
!$omp atomic update
   !ERROR: The atomic variable x should appear as an argument in the update operation
   x = y*1
!$omp atomic update
   !ERROR: The atomic variable x should appear as an argument in the update operation
   x = 1*y

!$omp atomic update
   x = x/1
!$omp atomic update
   x = 1/x
!$omp atomic update
   !ERROR: The atomic variable x should appear as an argument of the top-level / operator
   x = y/1
!$omp atomic update
   !ERROR: The atomic variable x should appear as an argument of the top-level / operator
   x = 1/y

!$omp atomic update
   m = m .AND. n
!$omp atomic update
   m = n .AND. m
!$omp atomic update
   !ERROR: The atomic variable m should appear as an argument of the top-level AND operator
   m = n .AND. l

!$omp atomic update
   m = m .OR. n
!$omp atomic update
   m = n .OR. m
!$omp atomic update
   !ERROR: The atomic variable m should appear as an argument of the top-level OR operator
   m = n .OR. l

!$omp atomic update
   m = m .EQV. n
!$omp atomic update
   m = n .EQV. m
!$omp atomic update
   !ERROR: The atomic variable m should appear as an argument of the top-level EQV operator
   m = n .EQV. l

!$omp atomic update
   m = m .NEQV. n
!$omp atomic update
   m = n .NEQV. m
!$omp atomic update
   !ERROR: The atomic variable m should appear as an argument of the top-level NEQV/EOR operator
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
        x = x

    !$omp atomic update
    !ERROR: This is not a valid ATOMIC UPDATE operation
        x = 1    

    !$omp atomic update
    !ERROR: The atomic variable a cannot be a proper subexpression of an argument (here: a*b) in the update operation
        a = a * b + a

    !$omp atomic
    !ERROR: The atomic variable a cannot be a proper subexpression of an argument (here: (a+9_4)) in the update operation
    !ERROR: The atomic variable a should appear as an argument of the top-level * operator
        a = b * (a + 9)

    !$omp atomic update
    !ERROR: The atomic variable a cannot be a proper subexpression of an argument (here: (a+b)) in the update operation
        a = a * (a + b)

    !$omp atomic
    !ERROR: The atomic variable a cannot be a proper subexpression of an argument (here: (b+a)) in the update operation
        a = (b + a) * a

    !$omp atomic
    !ERROR: The atomic variable a cannot be a proper subexpression of an argument (here: a*b) in the update operation
    !ERROR: The atomic variable a should appear as an argument of the top-level + operator
        a = a * b + c

    !This is expected to work due to reassociation.
    !$omp atomic update
        a = a + b + c

    !$omp atomic
        a = b * c + a

    !$omp atomic update
        a = c + b + a

    !$omp atomic
    !ERROR: No intrinsic or user-defined ASSIGNMENT(=) matches scalar INTEGER(4) and rank 1 array of INTEGER(4)
        a = a + d

    !$omp atomic update
    !ERROR: No intrinsic or user-defined ASSIGNMENT(=) matches scalar REAL(4) and rank 1 array of REAL(4)
    !ERROR: The atomic variable x cannot be a proper subexpression of an argument (here: x*y) in the update operation
    !ERROR: The atomic variable x should appear as an argument of the top-level / operator
        x = x * y / z

    !$omp atomic
    !ERROR: The atomic variable p%m should appear as an argument of the top-level + operator
        p%m = x + y

    !$omp atomic update
    !ERROR: No intrinsic or user-defined ASSIGNMENT(=) matches scalar REAL(4) and rank 1 array of REAL(4)
        p%m = p%m + p%n
end subroutine

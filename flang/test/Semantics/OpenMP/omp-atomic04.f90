! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp

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
   !ERROR: Atomic update variable 'x' not found in the RHS of the assignment statement in an ATOMIC (UPDATE) construct
   x = y + 1
!$omp atomic
   !ERROR: Atomic update variable 'x' not found in the RHS of the assignment statement in an ATOMIC (UPDATE) construct
   x = 1 + y

!$omp atomic
   x = x - 1
!$omp atomic
   x = 1 - x
!$omp atomic
   !ERROR: Atomic update variable 'x' not found in the RHS of the assignment statement in an ATOMIC (UPDATE) construct
   x = y - 1
!$omp atomic
   !ERROR: Atomic update variable 'x' not found in the RHS of the assignment statement in an ATOMIC (UPDATE) construct
   x = 1 - y

!$omp atomic
   x = x*1
!$omp atomic
   x = 1*x
!$omp atomic
   !ERROR: Atomic update variable 'x' not found in the RHS of the assignment statement in an ATOMIC (UPDATE) construct
   x = y*1
!$omp atomic
   !ERROR: Atomic update variable 'x' not found in the RHS of the assignment statement in an ATOMIC (UPDATE) construct
   x = 1*y

!$omp atomic
   x = x/1
!$omp atomic
   x = 1/x
!$omp atomic
   !ERROR: Atomic update variable 'x' not found in the RHS of the assignment statement in an ATOMIC (UPDATE) construct
   x = y/1
!$omp atomic
   !ERROR: Atomic update variable 'x' not found in the RHS of the assignment statement in an ATOMIC (UPDATE) construct
   x = 1/y

!$omp atomic
   m = m .AND. n
!$omp atomic
   m = n .AND. m
!$omp atomic
   !ERROR: Atomic update variable 'm' not found in the RHS of the assignment statement in an ATOMIC (UPDATE) construct
   m = n .AND. l

!$omp atomic
   m = m .OR. n
!$omp atomic
   m = n .OR. m
!$omp atomic
   !ERROR: Atomic update variable 'm' not found in the RHS of the assignment statement in an ATOMIC (UPDATE) construct
   m = n .OR. l

!$omp atomic
   m = m .EQV. n
!$omp atomic
   m = n .EQV. m
!$omp atomic
   !ERROR: Atomic update variable 'm' not found in the RHS of the assignment statement in an ATOMIC (UPDATE) construct
   m = n .EQV. l

!$omp atomic
   m = m .NEQV. n
!$omp atomic
   m = n .NEQV. m
!$omp atomic
   !ERROR: Atomic update variable 'm' not found in the RHS of the assignment statement in an ATOMIC (UPDATE) construct
   m = n .NEQV. l

!$omp atomic update
   x = x + 1
!$omp atomic update
   x = 1 + x
!$omp atomic update
   !ERROR: Atomic update variable 'x' not found in the RHS of the assignment statement in an ATOMIC (UPDATE) construct
   x = y + 1
!$omp atomic update
   !ERROR: Atomic update variable 'x' not found in the RHS of the assignment statement in an ATOMIC (UPDATE) construct
   x = 1 + y

!$omp atomic update
   x = x - 1
!$omp atomic update
   x = 1 - x
!$omp atomic update
   !ERROR: Atomic update variable 'x' not found in the RHS of the assignment statement in an ATOMIC (UPDATE) construct
   x = y - 1
!$omp atomic update
   !ERROR: Atomic update variable 'x' not found in the RHS of the assignment statement in an ATOMIC (UPDATE) construct
   x = 1 - y

!$omp atomic update
   x = x*1
!$omp atomic update
   x = 1*x
!$omp atomic update
   !ERROR: Atomic update variable 'x' not found in the RHS of the assignment statement in an ATOMIC (UPDATE) construct
   x = y*1
!$omp atomic update
   !ERROR: Atomic update variable 'x' not found in the RHS of the assignment statement in an ATOMIC (UPDATE) construct
   x = 1*y

!$omp atomic update
   x = x/1
!$omp atomic update
   x = 1/x
!$omp atomic update
   !ERROR: Atomic update variable 'x' not found in the RHS of the assignment statement in an ATOMIC (UPDATE) construct
   x = y/1
!$omp atomic update
   !ERROR: Atomic update variable 'x' not found in the RHS of the assignment statement in an ATOMIC (UPDATE) construct
   x = 1/y

!$omp atomic update
   m = m .AND. n
!$omp atomic update
   m = n .AND. m
!$omp atomic update
   !ERROR: Atomic update variable 'm' not found in the RHS of the assignment statement in an ATOMIC (UPDATE) construct
   m = n .AND. l

!$omp atomic update
   m = m .OR. n
!$omp atomic update
   m = n .OR. m
!$omp atomic update
   !ERROR: Atomic update variable 'm' not found in the RHS of the assignment statement in an ATOMIC (UPDATE) construct
   m = n .OR. l

!$omp atomic update
   m = m .EQV. n
!$omp atomic update
   m = n .EQV. m
!$omp atomic update
   !ERROR: Atomic update variable 'm' not found in the RHS of the assignment statement in an ATOMIC (UPDATE) construct
   m = n .EQV. l

!$omp atomic update
   m = m .NEQV. n
!$omp atomic update
   m = n .NEQV. m
!$omp atomic update
   !ERROR: Atomic update variable 'm' not found in the RHS of the assignment statement in an ATOMIC (UPDATE) construct
   m = n .NEQV. l

end program OmpAtomic

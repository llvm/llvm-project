! REQUIRES: openmp_runtime

! RUN: %python %S/../test_errors.py %s %flang_fc1 %openmp_flags

! OpenMP Atomic construct
! section 2.17.7
! operator is one of +, *, -, /, .AND., .OR., .EQV., or .NEQV

program OmpAtomic
   use omp_lib
   CHARACTER c*3, d*3
   LOGICAL l, m, n

   a = 1
   b = 2
   c = 'foo'
   d = 'bar'
   m = .TRUE.
   n = .FALSE.
   !$omp parallel num_threads(4)

   !$omp atomic
   a = a + (4*2)
   !$omp atomic
   a = a*(b + 1)
   !$omp atomic
   a = a - 3
   !$omp atomic
   a = a/(b + 1)
   !$omp atomic
   !ERROR: The ** operator is not a valid ATOMIC UPDATE operation
   a = a**4
   !$omp atomic 
   !ERROR: Atomic variable c cannot have CHARACTER type
   !ERROR: The atomic variable c should appear as an argument in the update operation
   c = d 
   !$omp atomic
   !ERROR: The < operator is not a valid ATOMIC UPDATE operation
   l = a .LT. b
   !$omp atomic
   !ERROR: The <= operator is not a valid ATOMIC UPDATE operation
   l = a .LE. b
   !$omp atomic
   !ERROR: The == operator is not a valid ATOMIC UPDATE operation
   l = a .EQ. b
   !$omp atomic
   !ERROR: The /= operator is not a valid ATOMIC UPDATE operation
   l = a .NE. b
   !$omp atomic
   !ERROR: The >= operator is not a valid ATOMIC UPDATE operation
   l = a .GE. b
   !$omp atomic
   !ERROR: The > operator is not a valid ATOMIC UPDATE operation
   l = a .GT. b
   !$omp atomic
   m = m .AND. n
   !$omp atomic
   m = m .OR. n
   !$omp atomic
   m = m .EQV. n
   !$omp atomic
   m = m .NEQV. n
   !$omp atomic update
   a = a + (4*2)
   !$omp atomic update
   a = a*(b + 1)
   !$omp atomic update
   a = a - 3
   !$omp atomic update
   a = a/(b + 1)
   !$omp atomic update
   !ERROR: The ** operator is not a valid ATOMIC UPDATE operation
   a = a**4
   !$omp atomic update
   !ERROR: Atomic variable c cannot have CHARACTER type
   !ERROR: This is not a valid ATOMIC UPDATE operation
   c = c//d
   !$omp atomic update
   !ERROR: The < operator is not a valid ATOMIC UPDATE operation
   l = a .LT. b
   !$omp atomic update
   !ERROR: The <= operator is not a valid ATOMIC UPDATE operation
   l = a .LE. b
   !$omp atomic update
   !ERROR: The == operator is not a valid ATOMIC UPDATE operation
   l = a .EQ. b
   !$omp atomic update
   !ERROR: The >= operator is not a valid ATOMIC UPDATE operation
   l = a .GE. b
   !$omp atomic update
   !ERROR: The > operator is not a valid ATOMIC UPDATE operation
   l = a .GT. b
   !$omp atomic update
   m = m .AND. n
   !$omp atomic update
   m = m .OR. n
   !$omp atomic update
   m = m .EQV. n
   !$omp atomic update
   m = m .NEQV. n
   !$omp end parallel
end program OmpAtomic

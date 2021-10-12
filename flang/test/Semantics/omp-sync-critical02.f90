! RUN: %python %S/test_errors.py %s %flang -fopenmp

! OpenMP Version 5.0
! 2.17.1 critical construct
! If the hint clause is specified, the critical construct must have a name.
program sample
   use omp_lib
   integer i, j
   !ERROR: Hint clause other than omp_sync_hint_none cannot be specified for an unnamed CRITICAL directive
   !$omp critical hint(omp_lock_hint_speculative)
   j = j + 1
   !$omp end critical

   !$omp critical (foo) hint(omp_lock_hint_speculative)
   i = i - 1
   !$omp end critical (foo)

   !ERROR: Hint clause other than omp_sync_hint_none cannot be specified for an unnamed CRITICAL directive
   !$omp critical hint(omp_lock_hint_nonspeculative)
   j = j + 1
   !$omp end critical

   !$omp critical (foo) hint(omp_lock_hint_nonspeculative)
   i = i - 1
   !$omp end critical (foo)

   !ERROR: Hint clause other than omp_sync_hint_none cannot be specified for an unnamed CRITICAL directive
   !$omp critical hint(omp_lock_hint_contended)
   j = j + 1
   !$omp end critical

   !$omp critical (foo) hint(omp_lock_hint_contended)
   i = i - 1
   !$omp end critical (foo)

   !ERROR: Hint clause other than omp_sync_hint_none cannot be specified for an unnamed CRITICAL directive
   !$omp critical hint(omp_lock_hint_uncontended)
   j = j + 1
   !$omp end critical

   !$omp critical (foo) hint(omp_lock_hint_uncontended)
   i = i - 1
   !$omp end critical (foo)
 
   !$omp critical hint(omp_sync_hint_none)
   j = j + 1
   !$omp end critical

   !$omp critical (foo) hint(omp_sync_hint_none)
   i = i - 1
   !$omp end critical (foo)

end program sample

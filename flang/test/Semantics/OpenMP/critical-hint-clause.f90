! REQUIRES: openmp_runtime

! RUN: %python %S/../test_errors.py %s %flang_fc1 %openmp_flags 
! Semantic checks on hint clauses, as they appear on critical construct

program sample
    use omp_lib
    integer :: y
    logical :: z
    real :: k
    integer :: p(1)
    
    !$omp critical (name) hint(1)
        y = 2
    !$omp end critical (name)
    
    !$omp critical (name) hint(2)
        y = 2
    !$omp end critical (name)
     
    !ERROR: The synchronization hint is not valid
    !$omp critical (name) hint(3)
        y = 2
    !$omp end critical (name)
    
    !$omp critical (name)  hint(5)
        y = 2
    !$omp end critical (name)
    
    !ERROR: The synchronization hint is not valid
    !$omp critical (name) hint(7)
        y = 2
    !$omp end critical (name)
   
    !ERROR: Synchronization hint must be a constant integer value
    !ERROR: Must be a constant value
    !$omp critical (name) hint(x)
        y = 2
    !$omp end critical (name)
    
    !$omp critical (name) hint(4)
        y = 2
    !$omp end critical (name)

    !$omp critical (name) hint(8)
        y = 2
    !$omp end critical (name)

    !$omp critical (name) hint(omp_sync_hint_uncontended)
        y = 2
    !$omp end critical (name)

    !$omp critical (name) hint(omp_lock_hint_speculative)
        y = 2
    !$omp end critical (name)
    
    !ERROR: Synchronization hint must be a constant integer value
    !ERROR: Must be a constant value
    !$omp critical (name) hint(omp_sync_hint_uncontended + omp_sync_hint) 
        y = 2
    !$omp end critical (name)

    !$omp critical (name) hint(omp_sync_hint_nonspeculative)
        y = 2
    !$omp end critical (name)

     !$omp critical (name) hint(omp_sync_hint_none)
        y = 2
    !$omp end critical (name)

    !$omp critical (name) hint(omp_sync_hint_uncontended + omp_lock_hint_speculative)
        y = 2
    !$omp end critical (name)

    !$omp critical (name) hint(omp_lock_hint_nonspeculative + omp_lock_hint_uncontended)
        y = 2
    !$omp end critical (name)

    !$omp critical (name) hint(omp_lock_hint_contended + omp_sync_hint_speculative)
        y = 2
    !$omp end critical (name)

    !$omp critical (name) hint(omp_lock_hint_contended + omp_sync_hint_nonspeculative)
        y = 2
    !$omp end critical (name)

    !ERROR: The synchronization hint is not valid
     !$omp critical (name) hint(omp_sync_hint_uncontended + omp_sync_hint_contended)
        y = 2
    !$omp end critical (name)

    !ERROR: The synchronization hint is not valid
    !$omp critical (name) hint(omp_sync_hint_nonspeculative + omp_lock_hint_speculative)
        y = 2
    !$omp end critical (name)

    !ERROR: Synchronization hint must be a constant integer value
    !ERROR: Must have INTEGER type, but is REAL(4)
    !$omp critical (name) hint(1.0) 
        y = 2
    !$omp end critical (name)

    !ERROR: Synchronization hint must be a constant integer value
    !ERROR: Operands of + must be numeric; have LOGICAL(4) and INTEGER(4)
    !$omp critical (name) hint(z + omp_sync_hint_nonspeculative)
        y = 2
    !$omp end critical (name)

    !ERROR: Synchronization hint must be a constant integer value
    !ERROR: Must be a constant value
    !$omp critical (name) hint(k + omp_sync_hint_speculative)
        y = 2
    !$omp end critical (name)

    !ERROR: Synchronization hint must be a constant integer value
    !ERROR: Must be a constant value
    !$omp critical (name) hint(p(1) + omp_sync_hint_uncontended)
        y = 2
    !$omp end critical (name)
end program


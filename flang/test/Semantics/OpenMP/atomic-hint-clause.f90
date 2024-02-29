! REQUIRES: openmp_runtime

! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp 
! Semantic checks on hint clauses, as they appear on atomic constructs

program sample
    use omp_lib
    integer :: x, y
    logical :: z
    real :: k
    integer :: p(1)
    integer, parameter :: a = 1
    !$omp atomic hint(1) write
        y = 2
    
    !$omp atomic read hint(2)
        y = x    
     
    !ERROR: Hint clause value is not a valid OpenMP synchronization value
    !$omp atomic hint(3)
        y = y + 10
    
    !$omp atomic update hint(5)
        y = x + y
    
    !ERROR: Hint clause value is not a valid OpenMP synchronization value
    !$omp atomic hint(7) capture
        y = x
        x = y
    !$omp end atomic
   
    !ERROR: Hint clause must have non-negative constant integer expression
    !ERROR: Must be a constant value
    !$omp atomic update hint(x)
        y = y * 1
    
    !$omp atomic read hint(4)
        y = x

    !$omp atomic hint(8)
        x = x * y

    !$omp atomic write hint(omp_sync_hint_uncontended)
        x = 10 * y

    !$omp atomic hint(omp_lock_hint_speculative)
        x = y + x
    
    !ERROR: Hint clause must have non-negative constant integer expression
    !ERROR: Must be a constant value
    !$omp atomic hint(omp_sync_hint_uncontended + omp_sync_hint) read
        y = x 

    !$omp atomic hint(omp_sync_hint_nonspeculative)
        y = y * 9

    !$omp atomic hint(omp_sync_hint_none) read
        y = x

    !$omp atomic read hint(omp_sync_hint_uncontended + omp_lock_hint_speculative)
        y = x

    !$omp atomic hint(omp_lock_hint_nonspeculative + omp_lock_hint_uncontended)
        x = x * y

    !$omp atomic write hint(omp_lock_hint_contended + omp_sync_hint_speculative)
        x = 10 * y

    !$omp atomic hint(omp_lock_hint_contended + omp_sync_hint_nonspeculative)
        x = y + x

    !ERROR: Hint clause value is not a valid OpenMP synchronization value
    !$omp atomic hint(omp_sync_hint_uncontended + omp_sync_hint_contended) read
        y = x 

    !ERROR: Hint clause value is not a valid OpenMP synchronization value
    !$omp atomic hint(omp_sync_hint_nonspeculative + omp_lock_hint_speculative)
        y = y * 9

    !ERROR: Hint clause must have non-negative constant integer expression
    !$omp atomic hint(1.0) read
        y = x

    !ERROR: Hint clause must have non-negative constant integer expression
    !ERROR: Operands of + must be numeric; have LOGICAL(4) and INTEGER(4)
    !$omp atomic hint(z + omp_sync_hint_nonspeculative) read
        y = x

    !ERROR: Hint clause must have non-negative constant integer expression
    !ERROR: Must be a constant value
    !$omp atomic hint(k + omp_sync_hint_speculative) read
        y = x

    !ERROR: Hint clause must have non-negative constant integer expression
    !ERROR: Must be a constant value
    !$omp atomic hint(p(1) + omp_sync_hint_uncontended) write
        x = 10 * y

    !$omp atomic write hint(a)
    !ERROR: RHS expression on atomic assignment statement cannot access 'x'
        x = y + x

    !$omp atomic hint(abs(-1)) write
        x = 7

    !$omp atomic hint(omp_sync_hint_uncontended + omp_sync_hint_uncontended + omp_sync_hint_speculative) write
        x = 7
end program

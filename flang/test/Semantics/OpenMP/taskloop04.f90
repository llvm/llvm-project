! When lowering Taskloop, it is possible for the TileSizes clause to be lowered, but this is not a supported clause.
! We should make sure that any use of Tilesizes with Taskloop is correctly rejected by the Semantics.
! RUN: %python %S/../test_errors.py %s %flang -fopenmp

subroutine test
    integer :: i, sum

    !ERROR: TILE cannot follow TASKLOOP
    !ERROR: SIZES clause is not allowed on the TASKLOOP directive
    !$omp taskloop tile sizes(2)
    do i=1,10
        sum = sum + i
    end do
    !$omp end taskloop
end subroutine

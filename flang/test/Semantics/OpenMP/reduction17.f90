! Test that Structure Component Array Elements are caught by Semantics and return an error
! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp -fopenmp-version=45

type test_type
    integer :: array(2)
end type

contains
    subroutine test
        type(test_type) :: x

        !ERROR: A variable that is part of another variable cannot appear on the REDUCTION clause
        !$omp do reduction(+: x%array(2))
        do i=1, 2
        end do
        !$omp end do
    end subroutine
end

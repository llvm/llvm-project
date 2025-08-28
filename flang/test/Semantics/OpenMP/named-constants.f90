!RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp

module named_constants
    implicit none
contains
    subroutine shrd()
        implicit none
        integer, parameter :: n = 7
        real, parameter :: m = 7.0
        logical, parameter :: l = .false.
        integer, dimension(3), parameter :: a = [1, 2, 3]
        ! no error expected
        !$omp parallel shared(n, m, l, a)
            print *, n, m, l, a
        !$omp end parallel
    end subroutine shrd

    subroutine frstprvt()
        implicit none
        integer, parameter :: n = 7
        real, parameter :: m = 7.0
        logical, parameter :: l = .false.
        integer, dimension(3), parameter :: a = [1, 2, 3]
        ! no error expected
        !$omp parallel firstprivate(n, m, l, a)
            print *, n, m, l, a
        !$omp end parallel
    end subroutine frstprvt

    subroutine prvt()
        implicit none
        integer, parameter :: n = 7
        real, parameter :: m = 7.0
        logical, parameter :: l = .false.
        integer, dimension(3), parameter :: a = [1, 2, 3]
        !ERROR: 'n' must be a variable
        !ERROR: 'm' must be a variable
        !ERROR: 'l' must be a variable
        !ERROR: 'a' must be a variable
        !$omp parallel private(n, m, l, a)
            print *, n, m, l, a
        !$omp end parallel
    end subroutine prvt
end module named_constants

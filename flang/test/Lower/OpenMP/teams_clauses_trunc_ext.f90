!RUN: %flang -fopenmp -c %s

module mod
    implicit none
contains
    subroutine num_teams_8(n)
        implicit none
        integer :: n, i
        !$omp target teams distribute num_teams(137_8)
        do i = 1, n
        end do
    end subroutine num_teams_8

    subroutine num_teams_2(n)
        implicit none
        integer :: n, i
        ! $omp target teams distribute num_teams(137_2)
        do i = 1, n
        end do
    end subroutine num_teams_2

    subroutine thread_limit_8(n)
        implicit none
        integer :: n, i
        ! $omp target teams distribute thread_limit(137_8)
        do i = 1, n
        end do
    end subroutine thread_limit_8

    subroutine thread_limit_2(n)
        implicit none
        integer :: n, i
        ! $omp target teams distribute thread_limit(137_2)
        do i = 1, n
        end do
    end subroutine thread_limit_2

    subroutine num_threads_8(n)
        implicit none
        integer :: n, i
        ! $omp target teams distribute parallel do num_threads(137_8)
        do i = 1, n
        end do
    end subroutine num_threads_8

    subroutine num_threads_2(n)
        implicit none
        integer :: n, i
        ! $omp target teams distribute parallel do num_threads(137_2)
        do i = 1, n
        end do
    end subroutine num_threads_2
end module mod

! RUN: %flang -o %t %s
! RUN: %t

program system_clock_test
    use iso_fortran_env, only: int64, real64
    implicit none

    integer, parameter :: delta = 1
    real(kind=real64), parameter :: epsilon = 0.001

    integer(kind=int64) :: t_start, t_end
    integer(kind=int64) :: rate
    real(kind=real64) :: diff

    call system_clock(count_rate=rate)

    call system_clock(t_start)
    call sleep(delta)
    call system_clock(t_end)

    diff = real(t_end - t_start, kind=real64) / real(rate, kind=real64)

    if (abs(diff - real(delta, kind=real64)) <= epsilon) then
        stop 0, quiet=.true.
    end if
    stop 1, quiet=.true.
end program system_clock_test

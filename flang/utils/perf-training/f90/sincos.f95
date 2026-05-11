! RUN: %flang -O3 -c %s
! RUN: %flang_skip_driver -O3 -c %s

program sincos_example
  implicit none
  integer, parameter :: size = 16
  integer :: i, max_iter
  real(8), dimension(size) :: dresult1, dresult2
  character(len = 32) :: arg

  max_iter = 10
  if (command_argument_count() .gt. 0) then
    call get_command_argument(1, arg)
    read(arg, *), max_iter
  end if
  if (max_iter .gt. size) stop
  do i = 1, max_iter
    dresult1(i) = dsin(dble(i))
    dresult2(i) = dcos(dble(i))
  end do
  do i = 1, max_iter
    print *, dresult1(i), dresult2(i)
  end do
end program

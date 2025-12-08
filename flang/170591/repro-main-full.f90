program main
  integer :: status_val

  ! Test case 1: status > 0, should trigger exit
  print *, "Calling KOHb_exit with status = 1"
  status_val = KOHb_exit(1)
  print *, "Returned from KOHb_exit (should not happen if exit is called): ", status_val

  ! Test case 2: status <= 0, should not trigger exit
  print *, "Calling KOHb_exit with status = 0"
  status_val = KOHb_exit(0)
  print *, "Returned from KOHb_exit: ", status_val

end program main

! The original function from the issue
integer function KOHb_exit(status)
  integer, intent(in) :: status
  if (status > 0) call exit(status) ! actually, _FortranAExit
  KOHb_exit = 0
  do i = 1, status
    print '(A,I0)', "KOHb #", i
  end do
end function KOHb_exit

integer function KOHb_exit(status)
  integer, intent(in) :: status
  if (status > 0) call exit(status) ! actually, _FortranAExit
  KOHb_exit = 0
  do i = 1, status
    print '(A,I0)', "KOHb #", i
  end do
end function KOHb_exit

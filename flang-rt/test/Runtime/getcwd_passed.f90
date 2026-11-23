! UNSUPPORTED: offload-cuda
! Issue: #223778
! [flang] getcwd() terminates string with c_null_char instead of blank padding
!
! RUN: %flang %isysroot -L"%libdir" %s -o %t
! RUN: env LD_LIBRARY_PATH="$LD_LIBRARY_PATH:%libdir" %t | FileCheck %s
!
! CHECK: PASS
program getcwd_padded
  implicit none
  character(len=4096) :: d
  integer :: st
  d = repeat(achar(0), len(d))
  call getcwd(d, st)
  if (st == 0 .and. index(d, achar(0)) == 0 .and. d(len(d):) == ' ' &
      .and. len_trim(d) > 0) then
    print *, 'PASS'
    stop
  end if
  print*, 'status =', st, ', expected to be: 0'
  print*, 'index(d, achar(0)) =', index(d, achar(0)), ', expected to be: 0'
  print*, "d(len(d):) == ' ' =", d(len(d):) == ' ', ', expected to be: .true.'
  print*, 'len_trim(d) =', len_trim(d), ', expected to be: > 0'
  print *, 'FAIL'
end program getcwd_padded

! UNSUPPORTED: offload-cuda

! Issue: #223778
! [flang] getcwd() terminates string with c_null_char instead of blank padding
!
! RUN: %flang %isysroot -L"%libdir" %s -o %t
! RUN: env LD_LIBRARY_PATH="$LD_LIBRARY_PATH:%libdir" %t | FileCheck %s
!

! CHECK: PASS

program getcwd_padded
  use iso_c_binding, only: c_null_char
  implicit none

  character(len=256) :: my_cwd
  integer :: idxNullByte

  call getcwd(my_cwd)

  idxNullByte = index(my_cwd, c_null_char)

  if (idxNullByte == 0) then
    print*, 'PASS'
  else
    print*, 'idxNullByte=', idxNullByte
    write(*,*), 'my_cwd=--->' // my_cwd // '<---'
    print*, 'FAIL'
  endif

end program getcwd_padded

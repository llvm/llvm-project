! REQUIRES: system-linux

! Verify that the hostname obtained by HOSTNM() intrinsic is the same
! as the hostname obtained by directly calling C gethostname().

! RUN: %flang -L"%libdir" %s -o %t
! RUN: env LD_LIBRARY_PATH="$LD_LIBRARY_PATH:%libdir" %t | FileCheck %s

! CHECK: PASS

program get_hostname_cinterop
  use, intrinsic :: iso_c_binding, only: c_char, c_int, c_size_t, c_null_char
  implicit none

  interface
    function gethostname(name, namelen) bind(C)
      import :: c_char, c_int, c_size_t
      integer(c_int) :: gethostname
      character(kind=c_char), dimension(*) :: name
      integer(c_size_t), value :: namelen
    end function gethostname
  end interface

  integer, parameter :: HOST_NAME_MAX = 255
  character(kind=c_char), dimension(HOST_NAME_MAX + 1) :: c_hostname
  character(HOST_NAME_MAX) :: hostname
  character(HOST_NAME_MAX) :: hostnm_str
  integer(c_int) :: status, i

  status = gethostname(c_hostname, HOST_NAME_MAX)
  if (status /= 0) then
    print *, "Error in gethostname(), status code: ", status
    error stop
  end if

  call hostnm(hostnm_str, status)
  if (status /= 0) then
    print *, "Error in hostnm(), status code: ", status
    error stop
  end if

  ! Find the position of the null terminator to convert C string to Fortran string
  i = 1
  do while (i <= HOST_NAME_MAX .and. c_hostname(i) /= c_null_char)
    i = i + 1
  end do

  hostname = transfer(c_hostname(1:i-1), hostname)

  print *, "Hostname from OS: ", hostname(1:i-1)
  print *, "Hostname from hostnm(): ", hostnm_str(1:i-1)

  if (hostname /= hostnm_str) then
    print *, "FAIL"
  else
    print *, "PASS"
  end if

end program get_hostname_cinterop


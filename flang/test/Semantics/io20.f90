! RUN: %python %S/test_errors.py %s %flang_fc1 -Werror
! RUN: %flang_fc1 -fsyntax-only -Wno-undelimited-namelist-character %s 2>&1 | FileCheck --allow-empty --check-prefix=DISABLED %s
! DISABLED-NOT: warning:
module m
  type :: t
    integer :: n
    character(4) :: s
  end type
  type :: dtio
    character(4) :: s
  contains
    procedure :: wf
    generic :: write(formatted) => wf
  end type
 contains
  subroutine wf(dtv, unit, iotype, vlist, iostat, iomsg)
    class(dtio), intent(in) :: dtv
    integer, intent(in) :: unit
    character(*), intent(in) :: iotype
    integer, intent(in) :: vlist(:)
    integer, intent(out) :: iostat
    character(*), intent(inout) :: iomsg
    write(unit, '(a)', iostat=iostat, iomsg=iomsg) dtv%s
  end subroutine
end module

subroutine s(d)
  use m
  character(*), intent(in) :: d
  character(1) :: c(3)
  character(10) :: buf
  integer :: i
  type(t) :: x
  type(dtio) :: y
  namelist /ncp/ c
  namelist /nint/ i
  namelist /nmix/ i, c
  namelist /nder/ x
  namelist /ndtio/ y
  !WARNING: CHARACTER output of namelist item 'c' might not be readable as namelist input if the delimiter mode is NONE; consider specifying DELIM='QUOTE' or DELIM='APOSTROPHE' [-Wundelimited-namelist-character]
  write(10, nml=ncp)
  !WARNING: CHARACTER output of namelist item 'c' might not be readable as namelist input if the delimiter mode is NONE; consider specifying DELIM='QUOTE' or DELIM='APOSTROPHE' [-Wundelimited-namelist-character]
  write(10, ncp)
  !WARNING: CHARACTER output of namelist item 'c' with DELIM='NONE' might not be readable as namelist input; consider specifying DELIM='QUOTE' or DELIM='APOSTROPHE' [-Wundelimited-namelist-character]
  write(10, nml=ncp, delim='none')
  !WARNING: CHARACTER output of namelist item 'c' might not be readable as namelist input if the delimiter mode is NONE; consider specifying DELIM='QUOTE' or DELIM='APOSTROPHE' [-Wundelimited-namelist-character]
  write(10, nml=nmix)
  !WARNING: CHARACTER output of namelist item 'x' might not be readable as namelist input if the delimiter mode is NONE; consider specifying DELIM='QUOTE' or DELIM='APOSTROPHE' [-Wundelimited-namelist-character]
  write(10, nml=nder)
  !WARNING: CHARACTER output of namelist item 'c' might not be readable as namelist input if the delimiter mode is NONE; consider specifying DELIM='QUOTE' or DELIM='APOSTROPHE' [-Wundelimited-namelist-character]
  write(buf, nml=ncp)
  !WARNING: CHARACTER output of namelist item 'c' with DELIM='NONE' might not be readable as namelist input; consider specifying DELIM='QUOTE' or DELIM='APOSTROPHE' [-Wundelimited-namelist-character]
  write(buf, nml=ncp, delim='none')
  !WARNING: CHARACTER output of namelist item 'c' might not be readable as namelist input if the delimiter mode is NONE; consider specifying DELIM='QUOTE' or DELIM='APOSTROPHE' [-Wundelimited-namelist-character]
  print ncp
  write(10, nml=ncp, delim='QUOTE')
  write(10, nml=ncp, delim='apostrophe')
  write(buf, nml=ncp, delim='quote')
  write(10, nml=ncp, delim=d)
  write(10, nml=nint)
  write(10, nml=ndtio)
  read(10, nml=ncp)
end

! An end of file in a list-directed READ with a derived-type item that has a
! defined formatted READ, and with neither END= nor IOSTAT=, terminates the
! program: after the item, and at the item itself.

! The runtime terminates with abort(). On Windows, abort() exits with status 3
! instead of raising a signal, so `not --crash` would not see a crash there.
! UNSUPPORTED: system-windows

! RUN: %flang %isysroot -L"%libdir" %s -o %t
! RUN: env LD_LIBRARY_PATH="$LD_LIBRARY_PATH:%libdir" not --crash %t after \
! RUN:   2>&1 | FileCheck %s
! RUN: env LD_LIBRARY_PATH="$LD_LIBRARY_PATH:%libdir" not --crash %t at \
! RUN:   2>&1 | FileCheck %s

! CHECK: fatal Fortran runtime error({{.*}}): End of file during input
! CHECK-NOT: not terminated

module dtio_list_input_eof_fatal_m
  implicit none
  type t
    integer :: n = -1
  contains
    procedure :: rd
    generic :: read(formatted) => rd
  end type
contains
  subroutine rd(dtv, unit, iotype, vlist, iostat, iomsg)
    class(t), intent(inout) :: dtv
    integer, intent(in) :: unit
    character(*), intent(in) :: iotype
    integer, intent(in) :: vlist(:)
    integer, intent(out) :: iostat
    character(*), intent(inout) :: iomsg
    read (unit, *, iostat=iostat, iomsg=iomsg) dtv%n
  end subroutine
end module

program dtio_list_input_eof_fatal
  use dtio_list_input_eof_fatal_m
  implicit none
  character(8) :: which
  character(4) :: line
  integer :: n
  type(t) :: x

  call get_command_argument(1, which)
  if (which == 'after') then
    line = '5'
    read (line, *) x, n
  else
    line = ' '
    read (line, *) x
  end if
  print '(a)', 'not terminated'
end program

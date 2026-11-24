! A list-directed READ that reaches the end of the file at a derived-type
! item with a defined formatted READ signals the end of file without calling
! the defined input procedure for that item.

! RUN: %flang %isysroot -L"%libdir" %s -o %t
! RUN: env LD_LIBRARY_PATH="$LD_LIBRARY_PATH:%libdir" not --crash %t 2>%t.err \
! RUN:   | FileCheck %s
! RUN: FileCheck --check-prefix=ERR %s < %t.err

! CHECK: IOSTAT=: T -1 0
! CHECK-NEXT: END= branch: -1 0
! CHECK-NEXT: array: T 7 -1 -1 1
! CHECK-NOT: not terminated

! ERR: fatal Fortran runtime error({{.*}}): End of file during input

module dtio_list_input_eof_m
  implicit none
  integer :: calls = 0
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
    calls = calls + 1
    read (unit, *, iostat=iostat, iomsg=iomsg) dtv%n
  end subroutine
end module

program dtio_list_input_eof
  use dtio_list_input_eof_m
  use iso_fortran_env, only: iostat_end
  implicit none
  character(4) :: line = ' '
  integer :: ios
  type(t) :: x, a(3)

  read (line, *, iostat=ios) x
  print '(a,l1,1x,i0,1x,i0)', 'IOSTAT=: ', ios == iostat_end, x%n, calls

  read (line, *, end=10) x
  print '(a)', 'no END= branch'
  goto 20
10 print '(a,i0,1x,i0)', 'END= branch: ', x%n, calls
20 continue

  line = '7'
  read (line, *, iostat=ios) a
  print '(a,l1,4(1x,i0))', 'array: ', ios == iostat_end, a%n, calls

  ! Neither END= nor IOSTAT=: the end of file terminates the program.
  line = ' '
  read (line, *) x
  print '(a)', 'not terminated'
end program

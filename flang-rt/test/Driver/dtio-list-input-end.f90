! After a derived-type item with a defined formatted READ, a list-directed
! READ statement keeps its END= and IOSTAT= handling. The case with neither
! END= nor IOSTAT= is in dtio-list-input-eof-fatal.f90.

! RUN: %flang %isysroot -L"%libdir" %s -o %t
! RUN: env LD_LIBRARY_PATH="$LD_LIBRARY_PATH:%libdir" %t | FileCheck %s

! CHECK: END= branch: 5
! CHECK-NEXT: IOSTAT=: T 5

module dtio_list_input_end_m
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

program dtio_list_input_end
  use dtio_list_input_end_m
  use iso_fortran_env, only: iostat_end
  implicit none
  character(4) :: line = '5'
  integer :: n, ios
  type(t) :: x

  read (line, *, end=10) x, n
  print '(a)', 'no END= branch'
  goto 20
10 print '(a,i0)', 'END= branch: ', x%n
20 continue

  x%n = -1
  read (line, *, iostat=ios) x, n
  print '(a,l1,1x,i0)', 'IOSTAT=: ', ios == iostat_end, x%n
end program

! Verify the iotype passed to a defined I/O procedure that is invoked, directly
! or indirectly, from a NAMELIST parent statement.  Only the procedure whose
! immediate parent is the namelist statement may see 'NAMELIST'; a procedure
! invoked from a list-directed child statement must see 'LISTDIRECTED'
! (F2023 12.6.4.8.3), at any nesting depth and in both directions.

! RUN: %flang %isysroot -L"%libdir" %s -o %t
! RUN: env LD_LIBRARY_PATH="$LD_LIBRARY_PATH:%libdir" %t | FileCheck %s

module iotype_nested_mod
  type :: leaf
    integer :: x
  contains
    procedure :: leaf_write
    procedure :: leaf_read
    generic :: write(formatted) => leaf_write
    generic :: read(formatted) => leaf_read
  end type

  type :: middle
    type(leaf) :: l
  contains
    procedure :: middle_write
    generic :: write(formatted) => middle_write
  end type

  type :: outer
    type(middle) :: m
  contains
    procedure :: outer_write
    generic :: write(formatted) => outer_write
  end type

  type :: reader
    type(leaf) :: l
  contains
    procedure :: reader_read
    generic :: read(formatted) => reader_read
  end type

  character(20) :: seen(3) = 'unset'

contains

  subroutine outer_write(dtv, unit, iotype, vlist, iostat, iomsg)
    class(outer), intent(in) :: dtv
    integer, intent(in) :: unit
    character(*), intent(in) :: iotype
    integer, intent(in) :: vlist(:)
    integer, intent(out) :: iostat
    character(*), intent(inout) :: iomsg
    seen(1) = iotype
    write (unit, *, iostat=iostat, iomsg=iomsg) dtv%m
  end subroutine

  subroutine middle_write(dtv, unit, iotype, vlist, iostat, iomsg)
    class(middle), intent(in) :: dtv
    integer, intent(in) :: unit
    character(*), intent(in) :: iotype
    integer, intent(in) :: vlist(:)
    integer, intent(out) :: iostat
    character(*), intent(inout) :: iomsg
    seen(2) = iotype
    write (unit, *, iostat=iostat, iomsg=iomsg) dtv%l
  end subroutine

  subroutine leaf_write(dtv, unit, iotype, vlist, iostat, iomsg)
    class(leaf), intent(in) :: dtv
    integer, intent(in) :: unit
    character(*), intent(in) :: iotype
    integer, intent(in) :: vlist(:)
    integer, intent(out) :: iostat
    character(*), intent(inout) :: iomsg
    seen(3) = iotype
    write (unit, *, iostat=iostat, iomsg=iomsg) dtv%x
  end subroutine

  subroutine reader_read(dtv, unit, iotype, vlist, iostat, iomsg)
    class(reader), intent(inout) :: dtv
    integer, intent(in) :: unit
    character(*), intent(in) :: iotype
    integer, intent(in) :: vlist(:)
    integer, intent(out) :: iostat
    character(*), intent(inout) :: iomsg
    seen(1) = iotype
    read (unit, *, iostat=iostat, iomsg=iomsg) dtv%l
  end subroutine

  subroutine leaf_read(dtv, unit, iotype, vlist, iostat, iomsg)
    class(leaf), intent(inout) :: dtv
    integer, intent(in) :: unit
    character(*), intent(in) :: iotype
    integer, intent(in) :: vlist(:)
    integer, intent(out) :: iostat
    character(*), intent(inout) :: iomsg
    seen(2) = iotype
    read (unit, *, iostat=iostat, iomsg=iomsg) dtv%x
  end subroutine

end module

program iotype_nested
  use iotype_nested_mod
  implicit none
  type(outer) :: o
  type(reader) :: r
  namelist /wgrp/ o
  namelist /rgrp/ r

  ! Three levels of defined output under a namelist WRITE.
  o%m%l%x = 9
  seen = 'unset'
  open (10, status='scratch')
  write (10, nml=wgrp)
  close (10)
  print *, 'write depth 1: ', trim(seen(1))
  print *, 'write depth 2: ', trim(seen(2))
  print *, 'write depth 3: ', trim(seen(3))
  ! CHECK: write depth 1: NAMELIST
  ! CHECK-NEXT: write depth 2: LISTDIRECTED
  ! CHECK-NEXT: write depth 3: LISTDIRECTED

  ! Two levels of defined input under a namelist READ.
  r%l%x = -1
  seen = 'unset'
  open (11, status='scratch')
  write (11, '(A)') '&RGRP R= 9/'
  rewind (11)
  read (11, nml=rgrp)
  close (11)
  print *, 'read depth 1: ', trim(seen(1))
  print *, 'read depth 2: ', trim(seen(2))
  print *, 'read value: ', r%l%x
  ! CHECK-NEXT: read depth 1: NAMELIST
  ! CHECK-NEXT: read depth 2: LISTDIRECTED
  ! CHECK-NEXT: read value: 9
end program

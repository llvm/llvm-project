! A defined I/O procedure may itself perform a child NAMELIST data transfer.
! Such a child statement is a namelist statement in its own right: a procedure
! it invokes must see iotype 'NAMELIST', and a child namelist READ must still
! be allowed to advance to further input records.  These properties must hold
! when the outer statement is a namelist statement too.

! RUN: %flang %isysroot -L"%libdir" %s -o %t
! RUN: env LD_LIBRARY_PATH="$LD_LIBRARY_PATH:%libdir" %t | FileCheck %s

module namelist_child_mod
  type :: leaf
    integer :: x
  contains
    procedure :: leaf_write
    generic :: write(formatted) => leaf_write
  end type

  type :: holder
    type(leaf) :: l
  contains
    procedure :: holder_write
    generic :: write(formatted) => holder_write
  end type

  type :: spanner
    integer :: total
  contains
    procedure :: spanner_read
    generic :: read(formatted) => spanner_read
  end type

  character(20) :: seen(2) = 'unset'
  type(leaf) :: inner_leaf
  namelist /inner_w/ inner_leaf
  integer :: a = -1, b = -1
  namelist /inner_r/ a, b

contains

  subroutine holder_write(dtv, unit, iotype, vlist, iostat, iomsg)
    class(holder), intent(in) :: dtv
    integer, intent(in) :: unit
    character(*), intent(in) :: iotype
    integer, intent(in) :: vlist(:)
    integer, intent(out) :: iostat
    character(*), intent(inout) :: iomsg
    seen(1) = iotype
    inner_leaf = dtv%l
    ! The child data transfer is itself a namelist statement.
    write (unit, nml=inner_w, iostat=iostat, iomsg=iomsg)
  end subroutine

  subroutine leaf_write(dtv, unit, iotype, vlist, iostat, iomsg)
    class(leaf), intent(in) :: dtv
    integer, intent(in) :: unit
    character(*), intent(in) :: iotype
    integer, intent(in) :: vlist(:)
    integer, intent(out) :: iostat
    character(*), intent(inout) :: iomsg
    seen(2) = iotype
    write (unit, *, iostat=iostat, iomsg=iomsg) dtv%x
  end subroutine

  subroutine spanner_read(dtv, unit, iotype, vlist, iostat, iomsg)
    class(spanner), intent(inout) :: dtv
    integer, intent(in) :: unit
    character(*), intent(in) :: iotype
    integer, intent(in) :: vlist(:)
    integer, intent(out) :: iostat
    character(*), intent(inout) :: iomsg
    seen(1) = iotype
    ! The child namelist group spans three records.
    read (unit, nml=inner_r, iostat=iostat, iomsg=iomsg)
    dtv%total = a + b
  end subroutine

end module

program namelist_child_io
  use namelist_child_mod
  implicit none
  type(holder) :: h
  type(spanner) :: s
  integer :: ios
  namelist /wgrp/ h
  namelist /rgrp/ s

  ! A child NAMELIST write nested in a namelist write.
  h%l%x = 9
  seen = 'unset'
  open (10, status='scratch')
  write (10, nml=wgrp)
  close (10)
  print *, 'outer iotype: ', trim(seen(1))
  print *, 'child-namelist iotype: ', trim(seen(2))
  ! CHECK: outer iotype: NAMELIST
  ! CHECK-NEXT: child-namelist iotype: NAMELIST

  ! A child NAMELIST read nested in a namelist read, spanning records.
  s%total = -1
  seen = 'unset'
  open (11, status='scratch')
  write (11, '(A)') '&RGRP S= &INNER_R'
  write (11, '(A)') ' A = 3'
  write (11, '(A)') ' B = 4 /'
  write (11, '(A)') ' /'
  rewind (11)
  ios = 0
  read (11, nml=rgrp, iostat=ios)
  close (11)
  print *, 'read iostat is zero: ', ios == 0
  print *, 'outer iotype: ', trim(seen(1))
  print *, 'child-namelist total: ', s%total
  ! CHECK-NEXT: read iostat is zero: T
  ! CHECK-NEXT: outer iotype: NAMELIST
  ! CHECK-NEXT: child-namelist total: 7
end program

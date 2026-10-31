! Both arms of the record-advancement rule for child list-directed input
! documented in flang/docs/Extensions.md: a non-NAMELIST list-directed child
! input statement may not advance to a further record when it has an ancestor
! formatted input statement that is not list-directed and there is no
! intervening NAMELIST, and may advance when such a NAMELIST intervenes.

! RUN: %flang %isysroot -L"%libdir" %s -o %t
! RUN: env LD_LIBRARY_PATH="$LD_LIBRARY_PATH:%libdir" %t | FileCheck %s

module child_advance_mod
  type :: pair
    integer :: p, q
  contains
    procedure :: pair_read
    generic :: read(formatted) => pair_read
  end type

  ! NAMELIST outside the formatted ancestor: no intervening NAMELIST.
  type :: blocked_outer
    type(pair) :: v
  contains
    procedure :: blocked_read
    generic :: read(formatted) => blocked_read
  end type

  ! NAMELIST below the formatted ancestor: the NAMELIST intervenes.
  type :: allowed_outer
    integer :: dummy
  contains
    procedure :: allowed_read
    generic :: read(formatted) => allowed_read
  end type

  type(pair) :: inner_pair
  namelist /inner/ inner_pair

contains

  subroutine blocked_read(dtv, unit, iotype, vlist, iostat, iomsg)
    class(blocked_outer), intent(inout) :: dtv
    integer, intent(in) :: unit
    character(*), intent(in) :: iotype
    integer, intent(in) :: vlist(:)
    integer, intent(out) :: iostat
    character(*), intent(inout) :: iomsg
    ! A formatted, non-list-directed child statement.
    read (unit, '(DT)', iostat=iostat, iomsg=iomsg) dtv%v
  end subroutine

  subroutine allowed_read(dtv, unit, iotype, vlist, iostat, iomsg)
    class(allowed_outer), intent(inout) :: dtv
    integer, intent(in) :: unit
    character(*), intent(in) :: iotype
    integer, intent(in) :: vlist(:)
    integer, intent(out) :: iostat
    character(*), intent(inout) :: iomsg
    ! A child NAMELIST statement below the formatted ancestor.
    read (unit, nml=inner, iostat=iostat, iomsg=iomsg)
    dtv%dummy = 1
  end subroutine

  subroutine pair_read(dtv, unit, iotype, vlist, iostat, iomsg)
    class(pair), intent(inout) :: dtv
    integer, intent(in) :: unit
    character(*), intent(in) :: iotype
    integer, intent(in) :: vlist(:)
    integer, intent(out) :: iostat
    character(*), intent(inout) :: iomsg
    ! The values sit on two different records, so this statement can only
    ! succeed if it is allowed to advance.
    read (unit, *, iostat=iostat, iomsg=iomsg) dtv%p, dtv%q
  end subroutine

end module

program namelist_child_advance
  use child_advance_mod
  implicit none
  type(blocked_outer) :: b
  type(allowed_outer) :: a
  integer :: ios
  namelist /grp/ b

  ! No intervening NAMELIST: the list-directed grandchild must not advance.
  b%v%p = -1
  b%v%q = -1
  open (10, status='scratch')
  write (10, '(A)') '&GRP B= 3'
  write (10, '(A)') ' 4 /'
  rewind (10)
  ios = 0
  read (10, nml=grp, iostat=ios)
  close (10)
  print *, 'no intervening namelist, advanced: ', ios == 0
  ! CHECK: no intervening namelist, advanced: F

  ! Intervening NAMELIST: the same grandchild must advance.
  inner_pair%p = -1
  inner_pair%q = -1
  open (11, status='scratch')
  write (11, '(A)') '&INNER INNER_PAIR= 3'
  write (11, '(A)') ' 4 /'
  rewind (11)
  ios = 0
  read (11, '(DT)', iostat=ios) a
  close (11)
  print *, 'intervening namelist, advanced: ', ios == 0
  print *, 'values: ', inner_pair%p, inner_pair%q
  ! CHECK-NEXT: intervening namelist, advanced: T
  ! CHECK-NEXT: values: 3 4
end program

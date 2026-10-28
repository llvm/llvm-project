! Verify that the value of iotype is correct when writing a derived type
! with a derived type component to a namelist.

! RUN: %flang %isysroot -L"%libdir" %s -o %t
! RUN: env LD_LIBRARY_PATH="$LD_LIBRARY_PATH:%libdir" %t

module iotype_mod
  type ty2
    integer :: x
  contains
    procedure :: type_2_prc
    generic :: write(formatted) => type_2_prc
  end type

  type ty1
    TYPE (ty2) :: t1
  contains
    procedure :: type_1_prc
    generic :: write(formatted) => type_1_prc
  end type

  character(14) :: ch_iotype_1='xxxxxxx'
  character(14) :: ch_iotype_2='yyyyyyy'
contains
  subroutine type_1_prc(dtv, unit, iotype, vlist, iostat, iomsg)
    class(ty1), intent(in) :: dtv
    integer, intent(in) :: unit
    character(*), intent(in) :: iotype
    integer, intent(in) :: vlist(:)
    integer, intent(out) :: iostat
    character(*), intent(inout) :: iomsg
    ch_iotype_1 =iotype
    write(unit,*,iostat=iostat,iomsg=iomsg) dtv%t1
  end subroutine

  subroutine type_2_prc(dtv, unit, iotype, vlist, iostat, iomsg)
    class(ty2), intent(in) :: dtv
    integer, intent(in) :: unit
    character(*), intent(in) :: iotype
    integer, intent(in) :: vlist(:)
    integer, intent(out) :: iostat
    character(*), intent(inout) :: iomsg
    ch_iotype_2 =iotype
    write(unit,*,iostat=iostat,iomsg=iomsg) dtv%x
  end subroutine
end module

program main
  USE iotype_mod
  type(ty1) :: obj
  namelist /NAME/obj
  obj%t1%x=9

  open(10,status="scratch")
  WRITE(10,NML=NAME)

  if (ch_iotype_1/='NAMELIST') error stop 1
  if (ch_iotype_2/='LISTDIRECTED') error stop 2
end program

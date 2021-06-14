! RUN: %S/test_errors.sh %s %t %flang_fc1
! REQUIRES: shell

! Tests for defined input/output.  See 12.6.4.8 and 15.4.3.2, and C777
module m1
  type,public :: t
    integer c
  contains
    procedure, nopass :: tbp=>formattedReadProc !Error, NOPASS not allowed
    !ERROR: Defined input/output procedure 'tbp' may not have NOPASS attribute
    generic :: read(formatted) => tbp
  end type
  private
contains
  subroutine formattedReadProc(dtv, unit, iotype, vlist, iostat, iomsg)
    class(t), intent(inout) :: dtv
    integer, intent(in) :: unit
    character(len=*), intent(in) :: iotype
    integer, intent(in) :: vlist(:)
    integer, intent(out) :: iostat
    character(len=*), intent(inout) :: iomsg

    iostat = 343
    stop 'fail'
  end subroutine
end module m1

module m2
  type,public :: t
    integer c
  contains
    procedure, pass :: tbp=>formattedReadProc
    !ERROR: Defined input/output procedure 'formattedreadproc' must have 6 dummy arguments rather than 5
    generic :: read(formatted) => tbp
  end type
  private
contains
  subroutine formattedReadProc(dtv, unit, iotype, vlist, iostat)
    class(t), intent(inout) :: dtv
    integer, intent(in) :: unit
    character(len=*), intent(in) :: iotype
    integer, intent(in) :: vlist(:)
    integer, intent(out) :: iostat

    iostat = 343
    stop 'fail'
  end subroutine
end module m2

module m3
  type,public :: t
    integer c
  contains
    procedure, pass :: tbp=>unformattedReadProc
    !ERROR: Defined input/output procedure 'unformattedreadproc' must have 4 dummy arguments rather than 5
    generic :: read(unformatted) => tbp
  end type
  private
contains
  ! Error bad # of args
  subroutine unformattedReadProc(dtv, unit, iostat, iomsg, iotype) 
    class(t), intent(inout) :: dtv
    integer, intent(in) :: unit
    integer, intent(out) :: iostat
    character(len=*), intent(inout) :: iomsg
    integer, intent(out) :: iotype

    iostat = 343
    stop 'fail'
  end subroutine
end module m3

module m4
  type,public :: t
    integer c
  contains
    procedure, pass :: tbp=>formattedReadProc
    generic :: read(formatted) => tbp
  end type
  private
contains
  !ERROR: Dummy argument 0 of 'formattedreadproc' must be a data object
  !ERROR: Cannot use an alternate return as the passed-object dummy argument
  subroutine formattedReadProc(*, unit, iotype, vlist, iostat, iomsg) 
    !ERROR: Dummy argument 'unit' must be a data object
    !ERROR: A dummy procedure without the POINTER attribute may not have an INTENT attribute
    procedure(sin), intent(in) :: unit
    character(len=*), intent(in) :: iotype
    integer, intent(in) :: vlist(:)
    integer, intent(out) :: iostat
    character(len=*), intent(inout) :: iomsg

    iostat = 343
    stop 'fail'
  end subroutine
end module m4

module m5
  type,public :: t
    integer c
  contains
    !ERROR: Passed-object dummy argument 'dtv' of procedure 'tbp' must be of type 't' but is 'INTEGER(4)'
    procedure, pass :: tbp=>formattedReadProc
    generic :: read(formatted) => tbp
  end type
  private
contains
  subroutine formattedReadProc(dtv, unit, iotype, vlist, iostat, iomsg)
    !ERROR: Dummy argument 'dtv' of a defined input/output procedure must have a derived type
    integer, intent(inout) :: dtv ! error, must be of type t
    integer, intent(in) :: unit
    character(len=*), intent(in) :: iotype
    integer, intent(in) :: vlist(:)
    integer, intent(out) :: iostat
    character(len=*), intent(inout) :: iomsg

    iostat = 343
    stop 'fail'
  end subroutine
end module m5

module m6
  interface read(formatted) 
    procedure :: formattedReadProc
  end interface

  contains
    subroutine formattedReadProc(dtv, unit, iotype, vlist, iostat, iomsg)
    !ERROR: Dummy argument 'dtv' of a defined input/output procedure must have a derived type
      integer, intent(inout) :: dtv
      integer, intent(in) :: unit
      character(len=*), intent(in) :: iotype ! error, must be deferred
      integer, intent(in) :: vlist(:)
      integer, intent(out) :: iostat
      character(len=*), intent(inout) :: iomsg
    end subroutine
end module m6

module m7
  type,public :: t
    integer c
  contains
    procedure, pass :: tbp=>formattedReadProc
    generic :: read(formatted) => tbp
  end type
  private
contains
  subroutine formattedReadProc(dtv, unit, iotype, vlist, iostat, iomsg)
    !ERROR: Dummy argument 'dtv' of a defined input/output procedure must have intent 'INTENT(INOUT)'
    class(t), intent(in) :: dtv ! Error, must be intent(inout)
    integer, intent(in) :: unit
    character(len=*), intent(in) :: iotype
    integer, intent(in) :: vlist(:)
    integer, intent(out) :: iostat
    character(len=*), intent(inout) :: iomsg

    iostat = 343
    stop 'fail'
  end subroutine
end module m7

module m8
  type,public :: t
    integer c
  contains
    procedure, pass :: tbp=>formattedWriteProc
    generic :: write(formatted) => tbp
  end type
  private
contains
  subroutine formattedWriteProc(dtv, unit, iotype, vlist, iostat, iomsg)
    !ERROR: Dummy argument 'dtv' of a defined input/output procedure must have intent 'INTENT(IN)'
    class(t), intent(inout) :: dtv ! Error, must be intent(inout)
    integer, intent(in) :: unit
    character(len=*), intent(in) :: iotype
    integer, intent(in) :: vlist(:)
    integer, intent(out) :: iostat
    character(len=*), intent(inout) :: iomsg

    iostat = 343
    stop 'fail'
  end subroutine
end module m8

module m9
  type,public :: t
    integer c
  contains
    procedure, pass :: tbp=>formattedReadProc
    generic :: read(formatted) => tbp
  end type
  private
contains
  subroutine formattedReadProc(dtv, unit, iotype, vlist, iostat, iomsg)
    class(t), intent(inout) :: dtv ! Error, can't have attributes
    !ERROR: Dummy argument 'unit' of a defined input/output procedure may not have any attributes
    integer,  pointer, intent(in) :: unit
    character(len=*), intent(in) :: iotype
    integer, intent(in) :: vlist(:) 
    integer, intent(out) :: iostat
    character(len=*), intent(inout) :: iomsg

    iostat = 343
    stop 'fail'
  end subroutine
end module m9

module m10
  type,public :: t
    integer c
  contains
    procedure, pass :: tbp=>formattedReadProc
    generic :: read(formatted) => tbp
  end type
  private
contains
  subroutine formattedReadProc(dtv, unit, iotype, vlist, iostat, iomsg)
    class(t), intent(inout) :: dtv
    !ERROR: Dummy argument 'unit' of a defined input/output procedure must be an INTEGER of default KIND
    real, intent(in) :: unit ! Error, must be an integer
    character(len=*), intent(in) :: iotype
    integer, intent(in) :: vlist(:)
    integer, intent(out) :: iostat
    character(len=*), intent(inout) :: iomsg

    iostat = 343
    stop 'fail'
  end subroutine
end module m10

module m11
  type,public :: t
    integer c
  contains
    procedure, pass :: tbp=>formattedReadProc
    generic :: read(formatted) => tbp
  end type
  private
contains
  subroutine formattedReadProc(dtv, unit, iotype, vlist, iostat, iomsg)
    class(t), intent(inout) :: dtv
    !ERROR: Dummy argument 'unit' of a defined input/output procedure must be an INTEGER of default KIND
    integer(8), intent(in) :: unit ! Error, must be default KIND
    character(len=*), intent(in) :: iotype
    integer, intent(in) :: vlist(:)
    integer, intent(out) :: iostat
    character(len=*), intent(inout) :: iomsg

    iostat = 343
    stop 'fail'
  end subroutine
end module m11

module m12
  type,public :: t
    integer c
  contains
    procedure, pass :: tbp=>formattedReadProc
    generic :: read(formatted) => tbp
  end type
  private
contains
  subroutine formattedReadProc(dtv, unit, iotype, vlist, iostat, iomsg)
    class(t), intent(inout) :: dtv
    !ERROR: Dummy argument 'unit' of a defined input/output procedure must be a scalar
    integer, dimension(22), intent(in) :: unit ! Error, must be a scalar
    character(len=*), intent(in) :: iotype
    integer, intent(in) :: vlist(:)
    integer, intent(out) :: iostat
    character(len=*), intent(inout) :: iomsg

    iostat = 343
    stop 'fail'
  end subroutine
end module m12

module m13
  type,public :: t
    integer c
  contains
    procedure, pass :: tbp=>formattedReadProc
    generic :: read(formatted) => tbp
  end type
  private
contains
  subroutine formattedReadProc(dtv, unit, iotype, vlist, iostat, iomsg)
    class(t), intent(inout) :: dtv
    !ERROR: Dummy argument 'unit' of a defined input/output procedure must have intent 'INTENT(IN)'
    integer, intent(out) :: unit !Error, must be intent(in)
    character(len=*), intent(in) :: iotype
    integer, intent(in) :: vlist(:)
    integer, intent(out) :: iostat
    character(len=*), intent(inout) :: iomsg

    iostat = 343
    stop 'fail'
  end subroutine
end module m13

module m14
  type,public :: t
    integer c
  contains
    procedure, pass :: tbp=>formattedReadProc
    generic :: read(formatted) => tbp
  end type
  private
contains
  subroutine formattedReadProc(dtv, unit, iotype, vlist, iostat, iomsg)
    class(t), intent(inout) :: dtv
    !ERROR: Dummy argument 'unit' of a defined input/output procedure must have intent 'INTENT(IN)'
    integer :: unit !Error, must be INTENT(IN)
    character(len=*), intent(in) :: iotype
    integer, intent(in) :: vlist(:)
    integer, intent(out) :: iostat
    character(len=*), intent(inout) :: iomsg

    iostat = 343
    stop 'fail'
  end subroutine
end module m14

module m15
  type,public :: t
    integer c
  contains
    procedure, pass :: tbp=>formattedReadProc
    generic :: read(formatted) => tbp
  end type
  private
contains
  subroutine formattedReadProc(dtv, unit, iotype, vlist, iostat, iomsg)
    class(t), intent(inout) :: dtv
    integer, intent(in) :: unit
    !ERROR: Dummy argument 'iotype' of a defined input/output procedure must be assumed-length CHARACTER
    character(len=5), intent(in) :: iotype ! Error, must be assumed length
    integer, intent(in) :: vlist(:)
    integer, intent(out) :: iostat
    character(len=*), intent(inout) :: iomsg

    iostat = 343
    stop 'fail'
  end subroutine
end module m15

module m16
  type,public :: t
    integer c
  contains
    procedure, pass :: tbp=>formattedReadProc
    generic :: read(formatted) => tbp
  end type
  private
contains
  subroutine formattedReadProc(dtv, unit, iotype, vlist, iostat, iomsg)
    class(t), intent(inout) :: dtv
    integer, intent(in) :: unit
    character(len=*), intent(in) :: iotype
    !ERROR: Dummy argument 'vlist' of a defined input/output procedure must be deferred shape
    integer, intent(in) :: vlist(5)
    integer, intent(out) :: iostat
    character(len=*), intent(inout) :: iomsg

    iostat = 343
    stop 'fail'
  end subroutine
end module m16

module m17
  ! Test the same defined input/output procedure specified as a generic
  type t
    integer c
  contains
    procedure :: formattedReadProc
  end type

  interface read(formatted)
    module procedure formattedReadProc
  end interface

contains
  subroutine formattedReadProc(dtv,unit,iotype,v_list,iostat,iomsg)
    class(t),intent(inout) :: dtv
    integer,intent(in) :: unit
    character(*),intent(in) :: iotype
    integer,intent(in) :: v_list(:)
    integer,intent(out) :: iostat
    character(*),intent(inout) :: iomsg
    read(unit,iotype,iostat=iostat,iomsg=iomsg) dtv%c
    print *,v_list
  end subroutine
end module

module m18
  ! Test the same defined input/output procedure specified as a type-bound 
  ! procedure and as a generic
  type t
    integer c
  contains
    procedure :: formattedReadProc
    generic :: read(formatted) => formattedReadProc
  end type
  interface read(formatted)
    module procedure formattedReadProc
  end interface
contains
  subroutine formattedReadProc(dtv,unit,iotype,v_list,iostat,iomsg)
    class(t),intent(inout) :: dtv
    integer,intent(in) :: unit
    character(*),intent(in) :: iotype
    integer,intent(in) :: v_list(:)
    integer,intent(out) :: iostat
    character(*),intent(inout) :: iomsg
    read(unit,iotype,iostat=iostat,iomsg=iomsg) dtv%c
    print *,v_list
  end subroutine
end module

module m19
  ! Test two different defined input/output procedures specified as a 
  ! type-bound procedure and as a generic for the same derived type
  type t
    integer c
  contains
    procedure :: unformattedReadProc1
    generic :: read(unformatted) => unformattedReadProc1
  end type
  interface read(unformatted)
    module procedure unformattedReadProc
  end interface
contains
  subroutine unformattedReadProc1(dtv,unit,iostat,iomsg)
    class(t),intent(inout) :: dtv
    integer,intent(in) :: unit
    integer,intent(out) :: iostat
    character(*),intent(inout) :: iomsg
    read(unit,iotype,iostat=iostat,iomsg=iomsg) dtv%c
    print *,v_list
  end subroutine
  !ERROR: Derived type 't' already has defined input/output procedure 'READUNFORMATTED'
  subroutine unformattedReadProc(dtv,unit,iostat,iomsg)
    class(t),intent(inout) :: dtv
    integer,intent(in) :: unit
    integer,intent(out) :: iostat
    character(*),intent(inout) :: iomsg
    read(unit,iotype,iostat=iostat,iomsg=iomsg) dtv%c
    print *,v_list
  end subroutine
end module

module m20
  ! Test read and write defined input/output procedures specified as a 
  ! type-bound procedure and as a generic for the same derived type
  type t
    integer c
  contains
    procedure :: unformattedReadProc
    generic :: read(unformatted) => unformattedReadProc
  end type
  interface read(unformatted)
    module procedure unformattedReadProc
  end interface
  interface write(unformatted)
    module procedure unformattedWriteProc
  end interface
contains
  subroutine unformattedReadProc(dtv,unit,iostat,iomsg)
    class(t),intent(inout) :: dtv
    integer,intent(in) :: unit
    integer,intent(out) :: iostat
    character(*),intent(inout) :: iomsg
    read(unit,iotype,iostat=iostat,iomsg=iomsg) dtv%c
    print *,v_list
  end subroutine
  subroutine unformattedWriteProc(dtv,unit,iostat,iomsg)
    class(t),intent(in) :: dtv
    integer,intent(in) :: unit
    integer,intent(out) :: iostat
    character(*),intent(inout) :: iomsg
    write(unit,iotype,iostat=iostat,iomsg=iomsg) dtv%c
    print *,v_list
  end subroutine
end module

module m21
  ! Test read and write defined input/output procedures specified as a 
  ! type-bound procedure and as a generic for the same derived type with a
  ! KIND type parameter where they both have the same value
  type t(typeParam)
    integer, kind :: typeParam = 4
    integer c
  contains
    procedure :: unformattedReadProc
    generic :: read(unformatted) => unformattedReadProc
  end type
  interface read(unformatted)
    module procedure unformattedReadProc1
  end interface
contains
  subroutine unformattedReadProc(dtv,unit,iostat,iomsg)
    class(t),intent(inout) :: dtv
    integer,intent(in) :: unit
    integer,intent(out) :: iostat
    character(*),intent(inout) :: iomsg
    read(unit,iotype,iostat=iostat,iomsg=iomsg) dtv%c
    print *,v_list
  end subroutine
  !ERROR: Derived type 't' already has defined input/output procedure 'READUNFORMATTED'
  subroutine unformattedReadProc1(dtv,unit,iostat,iomsg)
    class(t(4)),intent(inout) :: dtv
    integer,intent(in) :: unit
    integer,intent(out) :: iostat
    character(*),intent(inout) :: iomsg
    read(unit,iotype,iostat=iostat,iomsg=iomsg) dtv%c
    print *,v_list
  end subroutine
end module

module m22
  ! Test read and write defined input/output procedures specified as a 
  ! type-bound procedure and as a generic for the same derived type with a
  ! KIND type parameter where they have different values
  type t(typeParam)
    integer, kind :: typeParam = 4
    integer c
  contains
    procedure :: unformattedReadProc
    generic :: read(unformatted) => unformattedReadProc
  end type
  interface read(unformatted)
    module procedure unformattedReadProc1
  end interface
contains
  subroutine unformattedReadProc(dtv,unit,iostat,iomsg)
    class(t),intent(inout) :: dtv
    integer,intent(in) :: unit
    integer,intent(out) :: iostat
    character(*),intent(inout) :: iomsg
    read(unit,iotype,iostat=iostat,iomsg=iomsg) dtv%c
    print *,v_list
  end subroutine
  subroutine unformattedReadProc1(dtv,unit,iostat,iomsg)
    class(t(3)),intent(inout) :: dtv
    integer,intent(in) :: unit
    integer,intent(out) :: iostat
    character(*),intent(inout) :: iomsg
    read(unit,iotype,iostat=iostat,iomsg=iomsg) dtv%c
    print *,v_list
  end subroutine
end module

module m23
  type t(typeParam)
  ! Test read and write defined input/output procedures specified as a 
  ! type-bound procedure and as a generic for the same derived type with a
  ! LEN type parameter where they have different values
    integer, len :: typeParam = 4
    integer c
  contains
    procedure :: unformattedReadProc
    generic :: read(unformatted) => unformattedReadProc
  end type
  interface read(unformatted)
    module procedure unformattedReadProc1
  end interface
contains
  subroutine unformattedReadProc(dtv,unit,iostat,iomsg)
    class(t(*)),intent(inout) :: dtv
    integer,intent(in) :: unit
    integer,intent(out) :: iostat
    character(*),intent(inout) :: iomsg
    read(unit,iotype,iostat=iostat,iomsg=iomsg) dtv%c
    print *,v_list
  end subroutine
  subroutine unformattedReadProc1(dtv,unit,iostat,iomsg)
    class(t(3)),intent(inout) :: dtv
    integer,intent(in) :: unit
    integer,intent(out) :: iostat
    character(*),intent(inout) :: iomsg
    read(unit,iotype,iostat=iostat,iomsg=iomsg) dtv%c
    print *,v_list
  end subroutine
end module

module m24
  ! Test read and write defined input/output procedures specified as a 
  ! type-bound procedure and as a generic for the same derived type with a
  ! LEN type parameter where they have the same value
  type t(typeParam)
    integer, len :: typeParam = 4
    integer c
  contains
    procedure :: unformattedReadProc
    generic :: read(unformatted) => unformattedReadProc
  end type
  interface read(unformatted)
    module procedure unformattedReadProc1
  end interface
contains
  subroutine unformattedReadProc(dtv,unit,iostat,iomsg)
    class(t(*)),intent(inout) :: dtv
    integer,intent(in) :: unit
    integer,intent(out) :: iostat
    character(*),intent(inout) :: iomsg
    read(unit,iotype,iostat=iostat,iomsg=iomsg) dtv%c
    print *,v_list
  end subroutine
  !ERROR: Derived type 't' already has defined input/output procedure 'READUNFORMATTED'
  subroutine unformattedReadProc1(dtv,unit,iostat,iomsg)
    class(t(*)),intent(inout) :: dtv
    integer,intent(in) :: unit
    integer,intent(out) :: iostat
    character(*),intent(inout) :: iomsg
    read(unit,iotype,iostat=iostat,iomsg=iomsg) dtv%c
    print *,v_list
  end subroutine
end module

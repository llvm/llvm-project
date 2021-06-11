! RUN: %S/test_errors.sh %s %t %flang_fc1
! REQUIRES: shell
! NULL() intrinsic function error tests

subroutine test
  interface
    subroutine s0
    end subroutine
    subroutine s1(j)
      integer, intent(in) :: j
    end subroutine
    function f0()
      real :: f0
    end function
    function f1(x)
      real :: f1
      real, intent(inout) :: x
    end function
    function f2(p)
      import s0
      real :: f1
      procedure(s0), pointer, intent(inout) :: p
    end function
    function f3()
      import s1
      procedure(s1), pointer :: f3
    end function
  end interface
  type :: dt0
    integer, pointer :: ip0
  end type dt0
  type :: dt1
    integer, pointer :: ip1(:)
  end type dt1
  type :: dt2
    procedure(s0), pointer, nopass :: pps0
  end type dt2
  type :: dt3
    procedure(s1), pointer, nopass :: pps1
  end type dt3
  integer :: j
  type(dt0) :: dt0x
  type(dt1) :: dt1x
  type(dt2) :: dt2x
  type(dt3) :: dt3x
  integer, pointer :: ip0, ip1(:), ip2(:,:)
  integer, allocatable :: ia0, ia1(:), ia2(:,:)
  real, pointer :: rp0, rp1(:)
  integer, parameter :: ip0r = rank(null(mold=ip0))
  integer, parameter :: ip1r = rank(null(mold=ip1))
  integer, parameter :: ip2r = rank(null(mold=ip2))
  integer, parameter :: eight = ip0r + ip1r + ip2r + 5
  real(kind=eight) :: r8check
  ip0 => null() ! ok
  ip1 => null() ! ok
  ip2 => null() ! ok
  !ERROR: MOLD= argument to NULL() must be a pointer or allocatable
  ip0 => null(mold=1)
  !ERROR: MOLD= argument to NULL() must be a pointer or allocatable
  ip0 => null(mold=j)
  dt0x = dt0(null())
  dt0x = dt0(ip0=null())
  dt0x = dt0(ip0=null(ip0))
  dt0x = dt0(ip0=null(mold=ip0))
  !ERROR: function result type 'REAL(4)' is not compatible with pointer type 'INTEGER(4)'
  !ERROR: pointer 'ip0' is associated with the result of a reference to function 'null' whose pointer result has an incompatible type or shape
  dt0x = dt0(ip0=null(mold=rp0))
  !ERROR: function result type 'REAL(4)' is not compatible with pointer type 'INTEGER(4)'
  !ERROR: pointer 'ip1' is associated with the result of a reference to function 'null' whose pointer result has an incompatible type or shape
  dt1x = dt1(ip1=null(mold=rp1))
  dt2x = dt2(pps0=null())
  dt2x = dt2(pps0=null(mold=dt2x%pps0))
  !ERROR: Procedure pointer 'pps0' associated with result of reference to function 'null' that is an incompatible procedure pointer
  dt2x = dt2(pps0=null(mold=dt3x%pps1))
  !ERROR: Procedure pointer 'pps1' associated with result of reference to function 'null' that is an incompatible procedure pointer
  dt3x = dt3(pps1=null(mold=dt2x%pps0))
  dt3x = dt3(pps1=null(mold=dt3x%pps1))
end subroutine test

! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic
! NULL() intrinsic function error tests

subroutine test
  interface
    subroutine s0
    end subroutine
    subroutine s1(j)
      integer, intent(in) :: j
    end subroutine
    subroutine canbenull(x, y)
      integer, intent(in), optional :: x
      real, intent(in), pointer :: y
    end
    subroutine optionalAllocatable(x)
      integer, intent(in), allocatable, optional :: x
    end
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
  external implicit
  type :: dt0
    integer, pointer :: ip0
    integer :: n = 666
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
  type :: dt4
    real, allocatable :: ra0
  end type dt4
  type, extends(dt4) :: dt5
  end type dt5
  integer :: j
  type(dt0) :: dt0x
  type(dt1) :: dt1x
  type(dt2) :: dt2x
  type(dt3) :: dt3x
  type(dt4) :: dt4x
  integer, pointer :: ip0, ip1(:), ip2(:,:)
  integer, allocatable :: ia0, ia1(:), ia2(:,:)
  real, pointer :: rp0, rp1(:)
  integer, parameter :: ip0r = rank(null(mold=ip0))
  integer, parameter :: ip1r = rank(null(mold=ip1))
  integer, parameter :: ip2r = rank(null(mold=ip2))
  integer, parameter :: eight = ip0r + ip1r + ip2r + 5
  real(kind=eight) :: r8check
  logical, pointer :: lp
  type(dt4), pointer :: dt4p
  type(dt5), pointer :: dt5p
  ip0 => null() ! ok
  ip0 => null(null()) ! ok
  ip0 => null(null(null())) ! ok
  ip1 => null() ! ok
  ip1 => null(null()) ! ok
  ip1 => null(null(null())) ! ok
  ip2 => null() ! ok
  ip2 => null(null()) ! ok
  ip2 => null(null(null())) ! ok
  !ERROR: MOLD= argument to NULL() must be a pointer or allocatable
  ip0 => null(mold=1)
  !ERROR: MOLD= argument to NULL() must be a pointer or allocatable
  ip0 => null(null(mold=1))
  !ERROR: MOLD= argument to NULL() must be a pointer or allocatable
  ip0 => null(mold=j)
  !ERROR: MOLD= argument to NULL() must be a pointer or allocatable
  ip0 => null(mold=null(mold=j))
  dt0x = dt0(null())
  dt0x = dt0(ip0=null())
  dt0x = dt0(ip0=null(ip0))
  dt0x = dt0(ip0=null(mold=ip0))
  !ERROR: function result type 'REAL(4)' is not compatible with pointer type 'INTEGER(4)'
  dt0x = dt0(ip0=null(mold=rp0))
  !ERROR: A NULL pointer may not be used as the value for component 'n'
  dt0x = dt0(null(), null())
  !ERROR: function result type 'REAL(4)' is not compatible with pointer type 'INTEGER(4)'
  dt1x = dt1(ip1=null(mold=rp1))
  dt2x = dt2(pps0=null())
  dt2x = dt2(pps0=null(mold=dt2x%pps0))
  !ERROR: Procedure pointer 'pps0' associated with result of reference to function 'null' that is an incompatible procedure pointer: distinct numbers of dummy arguments
  dt2x = dt2(pps0=null(mold=dt3x%pps1))
  !ERROR: Procedure pointer 'pps1' associated with result of reference to function 'null' that is an incompatible procedure pointer: distinct numbers of dummy arguments
  dt3x = dt3(pps1=null(mold=dt2x%pps0))
  dt3x = dt3(pps1=null(mold=dt3x%pps1))
  dt4x = dt4(null()) ! ok
  !PORTABILITY: NULL() with arguments is not standard conforming as the value for allocatable component 'ra0' [-Wnull-mold-allocatable-component-value]
  dt4x = dt4(null(rp0))
  !PORTABILITY: NULL() with arguments is not standard conforming as the value for allocatable component 'ra0' [-Wnull-mold-allocatable-component-value]
  !ERROR: Rank-1 array value is not compatible with scalar component 'ra0'
  dt4x = dt4(null(rp1))
  !ERROR: A NULL procedure pointer may not be used as the value for component 'ra0'
  dt4x = dt4(null(dt2x%pps0))
  call canbenull(null(), null()) ! fine
  call canbenull(null(mold=ip0), null(mold=rp0)) ! fine
  !ERROR: ALLOCATABLE dummy argument 'x=' must be associated with an ALLOCATABLE actual argument
  call optionalAllocatable(null(mold=ip0))
  call optionalAllocatable(null(mold=ia0)) ! fine
  call optionalAllocatable(null()) ! fine
  !ERROR: Null pointer argument 'NULL()' requires an explicit interface
  call implicit(null())
  !ERROR: Null pointer argument 'null(mold=ip0)' requires an explicit interface
  call implicit(null(mold=ip0))
  !ERROR: A NULL() pointer is not allowed for 'x=' intrinsic argument
  print *, sin(null(rp0))
  !ERROR: A NULL() pointer is not allowed for 'x=' intrinsic argument
  print *, kind(null())
  print *, kind(null(rp0)) ! ok
  !ERROR: A NULL() pointer is not allowed for 'a=' intrinsic argument
  print *, extends_type_of(null(), null())
  print *, extends_type_of(null(dt5p), null(dt4p)) ! ok
  !ERROR: A NULL() pointer is not allowed for 'a=' intrinsic argument
  print *, same_type_as(null(), null())
  print *, same_type_as(null(dt5p), null(dt4p)) ! ok
  !ERROR: A NULL() pointer is not allowed for 'source=' intrinsic argument
  print *, transfer(null(rp0),ip0)
  !WARNING: Source of TRANSFER contains allocatable or pointer component %ra0 [-Wpointer-component-transfer-arg]
  print *, transfer(dt4(null()),[0])
  !ERROR: NULL() may not be used as an expression in this context
  select case(null(ip0))
  end select
  !ERROR: NULL() may not be used as an expression in this context
  if (null(lp)) then
  end if
end subroutine test

module m
  type :: pdt(n)
    integer, len :: n
  end type
 contains
  subroutine s1(x)
    character(*), pointer, intent(in) :: x
  end
  subroutine s2(x)
    type(pdt(*)), pointer, intent(in) :: x
  end
  subroutine s3(ar)
    real, pointer :: ar(..)
  end
  subroutine test(ar)
    real, pointer :: ar(..)
    !ERROR: Actual argument associated with dummy argument 'x=' is a NULL() pointer without a MOLD= to provide a character length
    call s1(null())
    !ERROR: Actual argument associated with dummy argument 'x=' is a NULL() pointer without a MOLD= to provide a value for the assumed type parameter 'n'
    call s2(null())
    !ERROR: MOLD= argument to NULL() must not be assumed-rank
    call s3(null(ar))
  end
end

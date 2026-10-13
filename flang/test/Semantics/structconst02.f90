! RUN: %python %S/test_errors.py %s %flang_fc1
! Error tests for structure constructors: per-component type
! (in)compatibility.

module module1
  interface
    real function realfunc(x)
      real, value :: x
    end function realfunc
  end interface
  type :: scalar(ik,rk,zk,ck,lk,len)
    integer, kind :: ik = 4, rk = 4, zk = 4, ck = 1, lk = 1
    integer, len :: len = 1
    integer(kind=ik) :: ix = int(0,kind=ik)
    real(kind=rk) :: rx = real(0.,kind=rk)
    complex(kind=zk) :: zx = cmplx(0.,0.,kind=zk)
    !ERROR: Initialization expression for 'cx' (" ") cannot be computed as a constant value
    character(kind=ck,len=len) :: cx = ' '
    logical(kind=lk) :: lx = .false.
    real(kind=rk), pointer :: rp => NULL()
    procedure(realfunc), pointer, nopass :: rfp1 => NULL()
    procedure(real), pointer, nopass :: rfp2 => NULL()
  end type scalar
 contains
  subroutine scalararg(x)
    type(scalar), intent(in) :: x
  end subroutine scalararg
  subroutine errors(n)
    integer, intent(in) :: n
    call scalararg(scalar(4)()) ! ok
    !ERROR: Structure constructor lacks a value for component 'cx'
    call scalararg(scalar(len=n)()) ! triggers error on 'cx'
    call scalararg(scalar(4)(ix=1,rx=2.,zx=(3.,4.),cx='a',lx=.true.))
    call scalararg(scalar(4)(1,2.,(3.,4.),'a',.true.))
!    call scalararg(scalar(4)(ix=5.,rx=6,zx=(7._8,8._2),cx=4_'b',lx=.true._4))
!    call scalararg(scalar(4)(5.,6,(7._8,8._2),4_'b',.true._4))
    call scalararg(scalar(4)(ix=5.,rx=6,zx=(7._8,8._2),cx=4_'b',lx=.true.))
    call scalararg(scalar(4)(5.,6,(7._8,8._2),4_'b',.true.))
    !ERROR: Value in structure constructor of type 'CHARACTER(KIND=1,LEN=1_8)' is incompatible with component 'ix' of type 'INTEGER(4)'
    call scalararg(scalar(4)(ix='a'))
    !ERROR: Value in structure constructor of type 'LOGICAL(4)' is incompatible with component 'ix' of type 'INTEGER(4)'
    call scalararg(scalar(4)(ix=.false.))
    !ERROR: Rank-1 array value is not compatible with scalar component 'ix'
    call scalararg(scalar(4)(ix=[1]))
    !TODO more!
  end subroutine errors
end module module1

! J3/26-007r1 C871 preserves protection in pointer components.
module protected_target_constructor
  type ordinary_box
    real, pointer :: p
  end type
  type protected_box
    real, pointer, protected_target :: p
  end type
  type ordinary_text_box
    character(:), pointer :: p
  end type
  type protected_text_box
    character(:), pointer, protected_target :: p
  end type
  type object
    real :: value
    real, pointer :: link
  end type
contains
  subroutine constructors
    real, pointer, protected_target :: p, a(:)
    real, pointer :: ordinary
    type(object), pointer, protected_target :: parent
    type(ordinary_box) :: writable
    type(protected_box) :: kept
    type(ordinary_text_box) :: writable_text
    type(protected_text_box) :: kept_text
    character(4), target :: text
    integer, pointer, protected_target :: index
    kept = protected_box(p)
    kept = protected_box(a(1))
    kept = protected_box(ordinary)
    kept = protected_box(null())
    writable = ordinary_box(ordinary)
    writable = ordinary_box(parent%link)
    !ERROR: Target 'p' with PROTECTED_TARGET attribute may not be associated with pointer 'p' without PROTECTED_TARGET
    writable = ordinary_box(p)
    !ERROR: Target 'a' with PROTECTED_TARGET attribute may not be associated with pointer 'p' without PROTECTED_TARGET
    writable = ordinary_box(a(1))
    !ERROR: Target 'parent' with PROTECTED_TARGET attribute may not be associated with pointer 'p' without PROTECTED_TARGET
    writable = ordinary_box(parent%value)
    associate(q => p)
      writable = ordinary_box(q) ! The associate has no PROTECTED_TARGET attribute.
    end associate
    writable_text = ordinary_text_box(text(index:index))
    kept_text = protected_text_box(text(index:index))
    associate(q => text(index:index))
      writable_text = ordinary_text_box(q)
      kept_text = protected_text_box(q)
    end associate
  end subroutine
end module

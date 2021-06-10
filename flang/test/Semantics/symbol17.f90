! RUN: %S/test_symbols.sh %s %t %flang_fc1
! REQUIRES: shell
! Forward references to derived types (non-error cases)

!DEF: /main MainProgram
program main
 !DEF: /main/t1 DerivedType
 type :: t1
  !DEF: /main/t2 DerivedType
  !DEF: /main/t1/t1a ALLOCATABLE ObjectEntity TYPE(t2)
  type(t2), allocatable :: t1a
  !REF: /main/t2
  !DEF: /main/t1/t1p POINTER ObjectEntity TYPE(t2)
  type(t2), pointer :: t1p
 end type
 !REF: /main/t2
 type :: t2
  !REF: /main/t2
  !DEF: /main/t2/t2a ALLOCATABLE ObjectEntity TYPE(t2)
  type(t2), allocatable :: t2a
  !REF: /main/t2
  !DEF: /main/t2/t2p POINTER ObjectEntity TYPE(t2)
  type(t2), pointer :: t2p
 end type
 !REF: /main/t1
 !DEF: /main/t1x TARGET ObjectEntity TYPE(t1)
 type(t1), target :: t1x
 !REF: /main/t1x
 !REF: /main/t1/t1a
 allocate(t1x%t1a)
 !REF: /main/t1x
 !REF: /main/t1/t1p
 !REF: /main/t1/t1a
 t1x%t1p => t1x%t1a
 !REF: /main/t1x
 !REF: /main/t1/t1a
 !REF: /main/t2/t2a
 allocate(t1x%t1a%t2a)
 !REF: /main/t1x
 !REF: /main/t1/t1a
 !REF: /main/t2/t2p
 !REF: /main/t2/t2a
 t1x%t1a%t2p => t1x%t1a%t2a
end program
!DEF: /f1/fwd DerivedType
!DEF: /f1 (Function) Subprogram TYPE(fwd)
!DEF: /f1/n (Implicit) ObjectEntity INTEGER(4)
type(fwd) function f1(n)
 !REF: /f1/fwd
 type :: fwd
  !DEF: /f1/fwd/n ObjectEntity INTEGER(4)
  integer :: n
 end type
 !DEF: /f1/f1 ObjectEntity TYPE(fwd)
 !REF: /f1/fwd/n
 !REF: /f1/n
 f1%n = n
end function
!DEF: /s1 (Subroutine) Subprogram
!DEF: /s1/q1 (Implicit) ObjectEntity TYPE(fwd)
subroutine s1 (q1)
 !DEF: /s1/fwd DerivedType
 implicit type(fwd)(q)
 !REF: /s1/fwd
 type :: fwd
  !DEF: /s1/fwd/n ObjectEntity INTEGER(4)
  integer :: n
 end type
 !REF: /s1/q1
 !REF: /s1/fwd/n
 q1%n = 1
end subroutine
!DEF: /f2/fwdpdt DerivedType
!DEF: /f2/kind INTRINSIC, PURE (Function) ProcEntity
!DEF: /f2 (Function) Subprogram TYPE(fwdpdt(k=4_4))
!DEF: /f2/n (Implicit) ObjectEntity INTEGER(4)
type(fwdpdt(kind(0))) function f2(n)
 !REF: /f2/fwdpdt
 !DEF: /f2/fwdpdt/k TypeParam INTEGER(4)
 type :: fwdpdt(k)
  !REF: /f2/fwdpdt/k
  integer, kind :: k
  !REF: /f2/fwdpdt/k
  !DEF: /f2/fwdpdt/n ObjectEntity INTEGER(int(int(k,kind=4),kind=8))
  integer(kind=k) :: n
 end type
 !DEF: /f2/f2 ObjectEntity TYPE(fwdpdt(k=4_4))
 !DEF: /f2/DerivedType2/n ObjectEntity INTEGER(4)
 !REF: /f2/n
 f2%n = n
end function
!DEF: /s2 (Subroutine) Subprogram
!DEF: /s2/q1 (Implicit) ObjectEntity TYPE(fwdpdt(k=4_4))
subroutine s2 (q1)
 !DEF: /s2/fwdpdt DerivedType
 !DEF: /s2/kind INTRINSIC, PURE (Function) ProcEntity
 implicit type(fwdpdt(kind(0)))(q)
 !REF: /s2/fwdpdt
 !DEF: /s2/fwdpdt/k TypeParam INTEGER(4)
 type :: fwdpdt(k)
  !REF: /s2/fwdpdt/k
  integer, kind :: k
  !REF: /s2/fwdpdt/k
  !DEF: /s2/fwdpdt/n ObjectEntity INTEGER(int(int(k,kind=4),kind=8))
  integer(kind=k) :: n
 end type
 !REF: /s2/q1
 !DEF: /s2/DerivedType2/n ObjectEntity INTEGER(4)
 q1%n = 1
end subroutine
!DEF: /m1 Module
module m1
 !DEF: /m1/forward PRIVATE DerivedType
  private :: forward
 !DEF: /m1/base PUBLIC DerivedType
  type :: base
  !REF: /m1/forward
  !DEF: /m1/base/p POINTER ObjectEntity CLASS(forward)
    class(forward), pointer :: p
  end type
 !REF: /m1/base
 !REF: /m1/forward
  type, extends(base) :: forward
  !DEF: /m1/forward/n ObjectEntity INTEGER(4)
    integer :: n
  end type
 contains
 !DEF: /m1/test PUBLIC (Subroutine) Subprogram
  subroutine test
  !REF: /m1/forward
  !DEF: /m1/test/object TARGET ObjectEntity TYPE(forward)
    type(forward), target :: object
  !REF: /m1/test/object
  !REF: /m1/base/p
    object%p => object
  !REF: /m1/test/object
  !REF: /m1/base/p
  !REF: /m1/forward/n
    object%p%n = 666
  end subroutine
end module

! RUN: %S/test_errors.sh %s %t %flang_fc1
! REQUIRES: shell
! Check for semantic errors in ALLOCATE statements

subroutine C945_a(srca, srcb, srcc, src_complex, src_logical, &
  srca2, srcb2, srcc2, src_complex2, srcx, srcx2)
! If type-spec appears, it shall specify a type with which each
! allocate-object is type compatible.

!second part C945, specific to SOURCE, is not checked here.

  type A
    integer i
  end type

  type, extends(A) :: B
    real, allocatable :: x(:)
  end type

  type, extends(B) :: C
    character(5) s
  end type

  type Unrelated
    class(A), allocatable :: polymorph
    type(A), allocatable :: notpolymorph
  end type

  real srcx, srcx2(6)
  class(A) srca, srca2(5)
  type(B) srcb, srcb2(6)
  class(C) srcc, srcc2(7)
  complex src_complex, src_complex2(8)
  complex src_logical(5)
  real, allocatable :: x1, x2(:)
  class(A), allocatable :: aa1, aa2(:)
  class(B), pointer :: bp1, bp2(:)
  class(C), allocatable :: ca1, ca2(:)
  class(*), pointer :: up1, up2(:)
  type(A), allocatable :: npaa1, npaa2(:)
  type(B), pointer :: npbp1, npbp2(:)
  type(C), allocatable :: npca1, npca2(:)
  class(Unrelated), allocatable :: unrelat

  allocate(x1, source=srcx)
  allocate(x2, mold=srcx2)
  allocate(bp2(3)%x, source=srcx2)
  !OK, type-compatible with A
  allocate(aa1, up1, unrelat%polymorph, unrelat%notpolymorph, &
    npaa1, source=srca)
  allocate(aa2, up2, npaa2, source=srca2)
  !OK, type compatible with B
  allocate(aa1, up1, unrelat%polymorph, bp1, npbp1, mold=srcb)
  allocate(aa2, up2, bp2, npbp2, mold=srcb2)
  !OK, type compatible with C
  allocate(aa1, up1, unrelat%polymorph, bp1, ca1, npca1, mold=srcc)
  allocate(aa2, up2, bp2, ca2, npca2, source=srcc2)


  !ERROR: Allocatable object in ALLOCATE must be type compatible with source expression from MOLD or SOURCE
  allocate(x1, mold=src_complex)
  !ERROR: Allocatable object in ALLOCATE must be type compatible with source expression from MOLD or SOURCE
  allocate(x2(2), source=src_complex2)
  !ERROR: Allocatable object in ALLOCATE must be type compatible with source expression from MOLD or SOURCE
  allocate(bp2(3)%x, mold=src_logical)
  !ERROR: Allocatable object in ALLOCATE must be type compatible with source expression from MOLD or SOURCE
  allocate(unrelat, mold=srca)
  !ERROR: Allocatable object in ALLOCATE must be type compatible with source expression from MOLD or SOURCE
  allocate(unrelat%notpolymorph, source=srcb)
  !ERROR: Allocatable object in ALLOCATE must be type compatible with source expression from MOLD or SOURCE
  allocate(npaa1, mold=srcb)
  !ERROR: Allocatable object in ALLOCATE must be type compatible with source expression from MOLD or SOURCE
  allocate(npaa2, source=srcb2)
  !ERROR: Allocatable object in ALLOCATE must be type compatible with source expression from MOLD or SOURCE
  allocate(npca1, bp1, npbp1, mold=srcc)
end subroutine

module m
  type :: t
    real x(100)
   contains
    procedure :: f
  end type
 contains
  function f(this) result (x)
    class(t) :: this
    class(t), allocatable :: x
  end function
  subroutine bar
    type(t) :: o
    type(t), allocatable :: p
    real, allocatable :: rp
    allocate(p, source=o%f())
    !ERROR: Allocatable object in ALLOCATE must be type compatible with source expression from MOLD or SOURCE
    allocate(rp, source=o%f())
  end subroutine
end module

! Related to C945, check typeless expression are caught

subroutine sub
end subroutine

function func() result(x)
  real :: x
end function

program test_typeless
  class(*), allocatable :: x
  interface
    subroutine sub
    end subroutine
    real function func()
    end function
  end interface
  procedure (sub), pointer :: subp => sub
  procedure (func), pointer :: funcp => func

  ! OK
  allocate(x, mold=func())
  allocate(x, source=funcp())

  !ERROR: Typeless item not allowed as SOURCE or MOLD in ALLOCATE
  allocate(x, mold=x'1')
  !ERROR: Typeless item not allowed as SOURCE or MOLD in ALLOCATE
  allocate(x, mold=sub)
  !ERROR: Typeless item not allowed as SOURCE or MOLD in ALLOCATE
  allocate(x, source=subp)
  !ERROR: Typeless item not allowed as SOURCE or MOLD in ALLOCATE
  allocate(x, mold=func)
  !ERROR: Typeless item not allowed as SOURCE or MOLD in ALLOCATE
  allocate(x, source=funcp)
end program

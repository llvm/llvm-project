! RUN: %S/test_errors.sh %s %t %f18
! Check for semantic errors in ALLOCATE statements


subroutine C934()
! If type-spec appears, it shall specify a type with which each
! allocate-object is type compatible.

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

  real, allocatable :: x1, x2(:)
  class(A), allocatable :: aa1, aa2(:)
  class(B), pointer :: bp1, bp2(:)
  class(C), allocatable :: ca1, ca2(:)
  class(*), pointer :: up1, up2(:)
  type(A), allocatable :: npaa1, npaa2(:)
  type(B), pointer :: npbp1, npbp2(:)
  type(C), allocatable :: npca1, npca2(:)
  class(Unrelated), allocatable :: unrelat

  allocate(real:: x1)
  allocate(real:: x2(2))
  allocate(real:: bp2(3)%x(5))
  !OK, type-compatible with A
  allocate(A:: aa1, aa2(2), up1, up2(3), &
    unrelat%polymorph, unrelat%notpolymorph, npaa1, npaa2(4))
  !OK, type compatible with B
  allocate(B:: aa1, aa2(2), up1, up2(3), &
    unrelat%polymorph, bp1, bp2(2), npbp1, npbp2(2:4))
  !OK, type compatible with C
  allocate(C:: aa1, aa2(2), up1, up2(3), &
    unrelat%polymorph, bp1, bp2(2), ca1, ca2(4), &
    npca1, npca2(2:4))


  !ERROR: Allocatable object in ALLOCATE must be type compatible with type-spec
  allocate(complex:: x1)
  !ERROR: Allocatable object in ALLOCATE must be type compatible with type-spec
  allocate(complex:: x2(2))
  !ERROR: Allocatable object in ALLOCATE must be type compatible with type-spec
  allocate(logical:: bp2(3)%x(5))
  !ERROR: Allocatable object in ALLOCATE must be type compatible with type-spec
  allocate(A:: unrelat)
  !ERROR: Allocatable object in ALLOCATE must be type compatible with type-spec
  allocate(B:: unrelat%notpolymorph)
  !ERROR: Allocatable object in ALLOCATE must be type compatible with type-spec
  allocate(B:: npaa1)
  !ERROR: Allocatable object in ALLOCATE must be type compatible with type-spec
  allocate(B:: npaa2(4))
  !ERROR: Allocatable object in ALLOCATE must be type compatible with type-spec
  allocate(C:: npca1, bp1, npbp1)
end subroutine

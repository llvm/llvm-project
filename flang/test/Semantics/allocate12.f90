! RUN: %S/test_errors.sh %s %t %f18
! Check for semantic errors in ALLOCATE statements

subroutine C941_C942b_C950(xsrc, x1, a2, b2, cx1, ca2, cb1, cb2, c1, c2)
! C941: An allocate-coarray-spec shall appear if and only if the allocate-object
! is a coarray.
  type type0
    real, allocatable :: array(:)
  end type
  type type1
    class(type0), pointer :: t0
  end type

  type type2
    type(type1), pointer :: t1(:)
  end type

  type A
    real x(10)
  end type

  type B
    real, allocatable :: x(:)
  end type

  type C
    class(type2), allocatable :: ct2(:, :)[:, :, :]
    class(A), allocatable :: cx(:, :)[:, :, :]
    class(A), allocatable :: x(:, :)
  end type

  real :: xsrc(10)
  real, allocatable :: x1, x2(:)
  class(A), pointer :: a1, a2(:)

  real, allocatable :: cx1[:], cx2(:)[:, :]
  class(A), allocatable :: ca1[:, :], ca2(:)[:]

  type(B) :: b1, b2(*)
  type(B) :: cb1[5:*], cb2(*)[2, -1:*]

  type(C) :: c1
  pointer :: c2(:, :)
  pointer :: varLocal(:)

  class(*), allocatable :: var(:), cvar(:)[:]

  ! Valid constructs
  allocate(x1, x2(10), cx1[*], cx2(10)[2, -1:*])
  allocate(a1, a2(10), ca1[2, -1:*], ca2(10)[*])
  allocate(b1%x, b2(1)%x, cb1%x, cb2(1)%x, SOURCE=xsrc)
  allocate(c1%x(-1:10, 1:5), c1%cx(-1:10, 1:5)[-1:5, 1:2, 2:*])
  allocate(c2(9, 27))
  allocate(varLocal(64))
  allocate(A:: var(5), cvar(10)[*])


  !ERROR: Coarray specification must not appear in ALLOCATE when allocatable object is not a coarray
  allocate(x1[*])
  !ERROR: Coarray specification must not appear in ALLOCATE when allocatable object is not a coarray
  allocate(x2(10)[*])
  !ERROR: Coarray specification must appear in ALLOCATE when allocatable object is a coarray
  allocate(cx1)
  !ERROR: Coarray specification must appear in ALLOCATE when allocatable object is a coarray
  allocate(cx2(10))

  !ERROR: Coarray specification must not appear in ALLOCATE when allocatable object is not a coarray
  allocate(cx1[*], a1[*])
  !ERROR: Coarray specification must not appear in ALLOCATE when allocatable object is not a coarray
  allocate(cx1[*], a2(10)[*])
  !ERROR: Coarray specification must appear in ALLOCATE when allocatable object is a coarray
  allocate(x1, ca1)
  !ERROR: Coarray specification must appear in ALLOCATE when allocatable object is a coarray
  allocate(ca1[2, -1:*], ca2(10))

  !ERROR: Coarray specification must not appear in ALLOCATE when allocatable object is not a coarray
  allocate(b1%x[5:*] , SOURCE=xsrc)
  !ERROR: Coarray specification must not appear in ALLOCATE when allocatable object is not a coarray
  allocate(b2(1)%x[2, -1:*], SOURCE=xsrc)
  !ERROR: Coarray specification must not appear in ALLOCATE when allocatable object is not a coarray
  allocate(cb1%x[5:*] , SOURCE=xsrc)
  !ERROR: Coarray specification must not appear in ALLOCATE when allocatable object is not a coarray
  allocate(cb2(1)%x[2, -1:*], SOURCE=xsrc)

  !ERROR: Coarray specification must not appear in ALLOCATE when allocatable object is not a coarray
  allocate(c1%x(-1:10, 1:5)[-1:5, 1:2, 2:*])
  !ERROR: Coarray specification must appear in ALLOCATE when allocatable object is a coarray
  allocate(c1%cx(-1:10, 1:5))

  !ERROR: Coarray specification must not appear in ALLOCATE when allocatable object is not a coarray
  allocate(A:: var(5)[*], cvar(10)[*])
  !ERROR: Coarray specification must appear in ALLOCATE when allocatable object is a coarray
  allocate(A:: var(5), cvar(10))

! C942b: [... (shape related stuff not tested here) ...]. The number of
! allocate-coshape-specs in an allocate-coarray-spec shall be one less
! than the corank of the allocate-object.

  ! Valid constructs already tested above

  !ERROR: Corank of coarray specification in ALLOCATE must match corank of alloctable coarray
  allocate(cx1[2,-1:*], cx2(10)[2, -1:*])
  !ERROR: Corank of coarray specification in ALLOCATE must match corank of alloctable coarray
  allocate(ca1[*], ca2(10)[*])
  !ERROR: Corank of coarray specification in ALLOCATE must match corank of alloctable coarray
  allocate(c1%cx(-1:10, 1:5)[-1:5, 1:*])
  !ERROR: Corank of coarray specification in ALLOCATE must match corank of alloctable coarray
  allocate(A:: cvar(10)[2,2,*])

! C950: An allocate-object shall not be a coindexed object.

  ! Valid construct
  allocate(c1%ct2(2,5)%t1(2)%t0%array(10))

  !ERROR: Allocatable object must not be coindexed in ALLOCATE
  allocate(b1%x, b2(1)%x, cb1[2]%x, SOURCE=xsrc)
  !ERROR: Allocatable object must not be coindexed in ALLOCATE
  allocate(b1%x, b2(1)%x, cb2(1)[2,-1]%x, MOLD=xsrc)
  !ERROR: Allocatable object must not be coindexed in ALLOCATE
  allocate(c1%ct2(2,5)[1,1,1]%t1(2)%t0%array(10))

end subroutine

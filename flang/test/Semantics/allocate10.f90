! RUN: %python %S/test_errors.py %s %flang_fc1
! Check for semantic errors in ALLOCATE statements

!TODO: mixing expr and source-expr?
!TODO: using subcomponent in source expressions

subroutine C939_C942a_C945b(xsrc1a, xsrc1c, xsrc0, xsrc2a, xsrc2c, pos)
! C939: If an allocate-object is an array, either allocate-shape-spec-list shall
! appear in its allocation, or source-expr shall appear in the ALLOCATE
! statement and have the same rank as the allocate-object.
  type A
    real, pointer :: x(:)
  end type
  real, allocatable :: x0
  real, allocatable :: x1(:)
  real, pointer :: x2(:, :, :)
  type(A) a1
  type(A), allocatable :: a2(:, :)

  real xsrc0
  real xsrc1a(*)
  real xsrc1b(2:7)
  real, pointer :: xsrc1c(:)
  real xsrc2a(4:8, 12, *)
  real xsrc2b(2:7, 5, 9)
  real, pointer :: xsrc2c(:, :, :)
  integer pos

  allocate(x1(5))
  allocate(x1(2:7))
  allocate(x1, SOURCE=xsrc1a(2:7))
  allocate(x1, MOLD=xsrc1b)
  allocate(x1, SOURCE=xsrc1c)

  allocate(x2(2,3,4))
  allocate(x2(2:7,3:8,4:9))
  allocate(x2, SOURCE=xsrc2a(4:8, 1:12, 2:5))
  allocate(x2, MOLD=cos(xsrc2b))
  allocate(x2, SOURCE=xsrc2c)

  allocate(x1(5), x2(2,3,4), a1%x(5), a2(1,2)%x(4))
  allocate(x1, a1%x, a2(1,2)%x, SOURCE=xsrc1a(2:7))
  allocate(x1, a1%x, a2(1,2)%x, MOLD=xsrc1b)
  allocate(x1, a1%x, a2(1,2)%x, SOURCE=xsrc1c)

  allocate(x0, x1(5), x2(2,3,4), a1%x(5), SOURCE=xsrc0)

  ! There are NO requirements that mold expression rank match the
  ! allocated-objects when allocate-shape-spec-lists are given.
  ! If it is not needed, the shape of MOLD should be simply ignored.
  allocate(x0, x1(5), x2(2,3,4), a1%x(5), MOLD=xsrc0)
  allocate(x0, x1(5), x2(2,3,4), a1%x(5), MOLD=xsrc1b)
  allocate(x0, x1(5), x2(2,3,4), a1%x(5), MOLD=xsrc2b)

  !ERROR: Arrays in ALLOCATE must have a shape specification or an expression of the same rank must appear in SOURCE or MOLD
  allocate(x1)
  !ERROR: Arrays in ALLOCATE must have a shape specification or an expression of the same rank must appear in SOURCE or MOLD
  allocate(x1, SOURCE=xsrc0)
  !ERROR: Arrays in ALLOCATE must have a shape specification or an expression of the same rank must appear in SOURCE or MOLD
  allocate(x1, MOLD=xsrc2c)

  !ERROR: Arrays in ALLOCATE must have a shape specification or an expression of the same rank must appear in SOURCE or MOLD
  allocate(x2, SOURCE=xsrc1a(2:7))
  !ERROR: Arrays in ALLOCATE must have a shape specification or an expression of the same rank must appear in SOURCE or MOLD
  allocate(x2, MOLD=xsrc1b)
  !ERROR: Arrays in ALLOCATE must have a shape specification or an expression of the same rank must appear in SOURCE or MOLD
  allocate(x2, SOURCE=xsrc1c)

  !ERROR: Arrays in ALLOCATE must have a shape specification or an expression of the same rank must appear in SOURCE or MOLD
  allocate(a1%x)
  !ERROR: Arrays in ALLOCATE must have a shape specification or an expression of the same rank must appear in SOURCE or MOLD
  allocate(a2(5,3)%x)
  !ERROR: Arrays in ALLOCATE must have a shape specification or an expression of the same rank must appear in SOURCE or MOLD
  allocate(x1(5), x2(2,3,4), a1%x, a2(1,2)%x(4))
  !ERROR: Arrays in ALLOCATE must have a shape specification or an expression of the same rank must appear in SOURCE or MOLD
  allocate(x2, a2(1,2)%x, SOURCE=xsrc2a(4:8, 1:12, 2:5))
  !ERROR: Arrays in ALLOCATE must have a shape specification or an expression of the same rank must appear in SOURCE or MOLD
  allocate(a1%x, MOLD=xsrc0)

 !C942a: The number of allocate-shape-specs in an allocate-shape-spec-list shall
 !be the same as the rank of the allocate-object. [...] (co-array stuffs).

 !ERROR: The number of shape specifications, when they appear, must match the rank of allocatable object
 allocate(x1(5, 5))
 !ERROR: The number of shape specifications, when they appear, must match the rank of allocatable object
 allocate(x1(2:3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 1, 2))
 !ERROR: The number of shape specifications, when they appear, must match the rank of allocatable object
 allocate(x2(pos))
 !ERROR: The number of shape specifications, when they appear, must match the rank of allocatable object
 allocate(x2(2, 3, pos+1, 5))
 !ERROR: The number of shape specifications, when they appear, must match the rank of allocatable object
 allocate(x1(5), x2(2,4), a1%x(5), a2(1,2)%x(4))

 !ERROR: The number of shape specifications, when they appear, must match the rank of allocatable object
 allocate(x1(2), a1%x(2,5), a2(1,2)%x(2))

 ! Test the check is not influenced by SOURCE
 !ERROR: The number of shape specifications, when they appear, must match the rank of allocatable object
 allocate(a1%x(5, 4, 3), SOURCE=xsrc2a(4:8, 1:4, 1:3))
 !ERROR: The number of shape specifications, when they appear, must match the rank of allocatable object
 allocate(x2(5), MOLD=xsrc1a(1:5))
 !ERROR: The number of shape specifications, when they appear, must match the rank of allocatable object
 allocate(a1%x(5, 4, 3), MOLD=xsrc1b)
 !ERROR: The number of shape specifications, when they appear, must match the rank of allocatable object
 allocate(x2(5), SOURCE=xsrc2b)

 ! C945b: If SOURCE= appears, source-expr shall be a scalar or have the same
 ! rank as each allocate-object.
 !ERROR: If SOURCE appears, the related expression must be scalar or have the same rank as each allocatable object in ALLOCATE
 allocate(x0, SOURCE=xsrc1b)
 !ERROR: If SOURCE appears, the related expression must be scalar or have the same rank as each allocatable object in ALLOCATE
 allocate(x2(2, 5, 7), SOURCE=xsrc1a(2:7))
 !ERROR: If SOURCE appears, the related expression must be scalar or have the same rank as each allocatable object in ALLOCATE
 allocate(x2(2, 5, 7), SOURCE=xsrc1c)

 !ERROR: If SOURCE appears, the related expression must be scalar or have the same rank as each allocatable object in ALLOCATE
 allocate(x1(5), SOURCE=xsrc2a(4:8, 1:12, 2:5))
 !ERROR: If SOURCE appears, the related expression must be scalar or have the same rank as each allocatable object in ALLOCATE
 allocate(x1(3), SOURCE=cos(xsrc2b))
 !ERROR: If SOURCE appears, the related expression must be scalar or have the same rank as each allocatable object in ALLOCATE
 allocate(x1(100), SOURCE=xsrc2c)

 !ERROR: If SOURCE appears, the related expression must be scalar or have the same rank as each allocatable object in ALLOCATE
 allocate(a1%x(10), x2(20, 30, 40), a2(1,2)%x(50), SOURCE=xsrc1c)
 !ERROR: If SOURCE appears, the related expression must be scalar or have the same rank as each allocatable object in ALLOCATE
 allocate(a1%x(25), SOURCE=xsrc2b)

end subroutine

subroutine C940(a1, pos)
! If allocate-object is scalar, allocate-shape-spec-list shall not appear.
  type A
    integer(kind=8), allocatable :: i
  end type

  type B(k, l1, l2, l3)
    integer, kind :: k
    integer, len :: l1, l2, l3
    real(kind=k) x(-1:l1, 0:l2, 1:l3)
  end type

  integer pos
  class(A), allocatable :: a1(:)
  real, pointer :: x
  type(B(8,4,5,6)), allocatable :: b1

  ! Nominal
  allocate(x)
  allocate(a1(pos)%i)
  allocate(b1)

  !ERROR: Shape specifications must not appear when allocatable object is scalar
  allocate(x(pos))
  !ERROR: Shape specifications must not appear when allocatable object is scalar
  allocate(a1(pos)%i(5:2))
  !ERROR: Shape specifications must not appear when allocatable object is scalar
  allocate(b1(1))
end subroutine

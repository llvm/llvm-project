! RUN: %S/test_errors.sh %s %t %f18
! Check for semantic errors in ALLOCATE statements


subroutine C933_b(n)
! If any allocate-object has a deferred type parameter, is unlimited polymorphic,
! or is of abstract type, either type-spec or source-expr shall appear.

! only testing unlimited polymorphic and abstract-type here

  type, abstract :: Base
    integer x
  end type

  type, extends(Base) :: A
    integer y
  end type

  type, extends(Base) :: B
    class(Base), allocatable :: y
  end type

  type C
    class(*), pointer :: whatever
    real, pointer :: y
  end type

  integer n
  class(*), allocatable :: u1, u2(:)
  class(C), allocatable :: n1, n2(:)
  class(Base), pointer :: p1, p2(:)
  class(B), pointer :: p3, p4(:)
  type(A) :: molda = A(1, 2)

  !ERROR: Either type-spec or source-expr must appear in ALLOCATE when allocatable object is unlimited polymorphic
  allocate(u1)
  !ERROR: Either type-spec or source-expr must appear in ALLOCATE when allocatable object is unlimited polymorphic
  allocate(u2(2))
  !ERROR: Either type-spec or source-expr must appear in ALLOCATE when allocatable object is unlimited polymorphic
  allocate(n1%whatever)
  !ERROR: Either type-spec or source-expr must appear in ALLOCATE when allocatable object is unlimited polymorphic
  allocate(n2(2)%whatever)
  !ERROR: Either type-spec or source-expr must appear in ALLOCATE when allocatable object is of abstract type
  allocate(p1)
  !ERROR: Either type-spec or source-expr must appear in ALLOCATE when allocatable object is of abstract type
  allocate(p2(2))
  !ERROR: Either type-spec or source-expr must appear in ALLOCATE when allocatable object is of abstract type
  allocate(p3%y)
  !ERROR: Either type-spec or source-expr must appear in ALLOCATE when allocatable object is of abstract type
  allocate(p4(2)%y)
  !WRONG allocate(Base:: u1)

  ! No error expected
  allocate(real:: u1, u2(2))
  allocate(A:: u1, u2(2))
  allocate(C:: u1, u2(2))
  allocate(character(n):: u1, u2(2))
  allocate(C:: n1%whatever, n2(2)%whatever)
  allocate(A:: p1, p2(2))
  allocate(B:: p3%y, p4(2)%y)
  allocate(u1, u2(2), MOLD = cos(5.+n))
  allocate(u1, u2(2), MOLD = molda)
  allocate(u1, u2(2), MOLD = n1)
  allocate(u1, u2(2), MOLD = new_line("a"))
  allocate(n1%whatever, MOLD = n2(1))
  allocate(p1, p2(2), MOLD = p3)
  allocate(p3%y, p4(2)%y, MOLD = B(5))
  allocate(u1, u2(2), SOURCE = cos(5.+n))
  allocate(u1, u2(2), SOURCE = molda)
  allocate(u1, u2(2), SOURCE = n1)
  allocate(u1, u2(2), SOURCE = new_line("a"))
  allocate(n1%whatever, SOURCE = n2(1))
  allocate(p1, p2(2), SOURCE = p3)
  allocate(p3%y, p4(2)%y, SOURCE = B(5))

  ! OK, not unlimited polymorphic or abstract
  allocate(n1, n2(2))
  allocate(p3, p4(2))
end subroutine

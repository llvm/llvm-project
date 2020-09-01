! RUN: %S/test_errors.sh %s %t %f18
! Initializer error tests

subroutine objectpointers(j)
  integer, intent(in) :: j
  real, allocatable, target, save :: x1
  real, codimension[*], target, save :: x2
  real, save :: x3
  real, target :: x4
  real, target, save :: x5(10)
!ERROR: An initial data target may not be a reference to an ALLOCATABLE 'x1'
  real, pointer :: p1 => x1
!ERROR: An initial data target may not be a reference to a coarray 'x2'
  real, pointer :: p2 => x2
!ERROR: An initial data target may not be a reference to an object 'x3' that lacks the TARGET attribute
  real, pointer :: p3 => x3
!ERROR: An initial data target may not be a reference to an object 'x4' that lacks the SAVE attribute
  real, pointer :: p4 => x4
!ERROR: An initial data target must be a designator with constant subscripts
  real, pointer :: p5 => x5(j)
!ERROR: Pointer has rank 0 but target has rank 1
  real, pointer :: p6 => x5

!TODO: type incompatibility, non-deferred type parameter values, contiguity

end subroutine

subroutine dataobjects(j)
  integer, intent(in) :: j
  real, parameter :: x1(*) = [1., 2.]
!ERROR: Implied-shape array 'x2' has rank 2, but its initializer has rank 1
  real, parameter :: x2(*,*) = [1., 2.]
!ERROR: Shape of 'x3' is not implied, deferred, nor constant
  real, parameter :: x3(j) = [1., 2.]
!ERROR: An automatic variable must not be initialized
  real :: x4(j) = [1., 2.]
!ERROR: 'x5' has rank 2, but its initializer has rank 1
  real :: x5(2,2) = [1., 2., 3., 4.]
  real :: x6(2,2) = 5.
!ERROR: 'x7' has rank 0, but its initializer has rank 1
  real :: x7 = [1.]
  real :: x8(2,2) = reshape([1., 2., 3., 4.], [2, 2])
!ERROR: Dimension 1 of object has extent 3, but initializer has extent 2
  real :: x9(3) = [1., 2.]
!ERROR: Dimension 1 of object has extent 2, but initializer has extent 3
  real :: x10(2,3) = reshape([real::(k,k=1,6)], [3, 2])
end subroutine

subroutine components
  real, target, save :: a1(3)
  real, target :: a2
  real, save :: a3
  real, target, save :: a4
  type :: t1
!ERROR: Dimension 1 of object has extent 2, but initializer has extent 3
    real :: x1(2) = [1., 2., 3.]
  end type
  type :: t2(kind, len)
    integer, kind :: kind
    integer, len :: len
    real :: x1(2) = [1., 2., 3.]
    real :: x2(kind) = [1., 2., 3.]
    real :: x3(len) = [1., 2., 3.]
    real, pointer :: p1(:) => a1
!ERROR: An initial data target may not be a reference to an object 'a2' that lacks the SAVE attribute
    real, pointer :: p2 => a2
!ERROR: An initial data target may not be a reference to an object 'a3' that lacks the TARGET attribute
    real, pointer :: p3 => a3
!ERROR: Pointer has rank 0 but target has rank 1
    real, pointer :: p4 => a1
!ERROR: Pointer has rank 1 but target has rank 0
    real, pointer :: p5(:) => a4
  end type
!ERROR: Dimension 1 of object has extent 2, but initializer has extent 3
  type(t2(3,3)) :: o1
!ERROR: Dimension 1 of object has extent 2, but initializer has extent 3
!ERROR: Dimension 1 of object has extent 2, but initializer has extent 3
!ERROR: Dimension 1 of object has extent 2, but initializer has extent 3
  type(t2(2,2)) :: o2
end subroutine

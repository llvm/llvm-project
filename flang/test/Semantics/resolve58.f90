! RUN: %S/test_errors.sh %s %t %flang_fc1
subroutine s1(x, y)
  !ERROR: Array pointer 'x' must have deferred shape or assumed rank
  real, pointer :: x(1:)  ! C832
  !ERROR: Allocatable array 'y' must have deferred shape or assumed rank
  real, dimension(1:,1:), allocatable :: y  ! C832
end

subroutine s2(a, b, c)
  real :: a(:,1:)
  real :: b(10,*)
  real :: c(..)
  !ERROR: Array pointer 'd' must have deferred shape or assumed rank
  real, pointer :: d(:,1:)  ! C832
  !ERROR: Allocatable array 'e' must have deferred shape or assumed rank
  real, allocatable :: e(10,*)  ! C832
  !ERROR: Assumed-rank array 'f' must be a dummy argument
  real, pointer :: f(..)  ! C837
  !ERROR: Assumed-shape array 'g' must be a dummy argument
  real :: g(:,1:)
  !ERROR: Assumed-size array 'h' must be a dummy argument
  real :: h(10,*)  ! C833
  !ERROR: Assumed-rank array 'i' must be a dummy argument
  real :: i(..)  ! C837
end

subroutine s3(a, b)
  real :: a(*)
  !ERROR: Dummy array argument 'b' may not have implied shape
  real :: b(*,*)  ! C836
  !ERROR: Implied-shape array 'c' must be a named constant
  real :: c(*)  ! C836
  !ERROR: Named constant 'd' array must have constant or implied shape
  integer, parameter :: d(:) = [1, 2, 3]
end

subroutine s4()
  type :: t
    integer, allocatable :: a(:)
    !ERROR: Component array 'b' without ALLOCATABLE or POINTER attribute must have explicit shape
    integer :: b(:)  ! C749
    real, dimension(1:10) :: c
    !ERROR: Array pointer component 'd' must have deferred shape
    real, pointer, dimension(1:10) :: d  ! C745
  end type
end

function f()
  !ERROR: Array 'f' without ALLOCATABLE or POINTER attribute must have explicit shape
  real, dimension(:) :: f  ! C832
end

subroutine s5()
  !ERROR: Allocatable array 'a' must have deferred shape or assumed rank
  integer :: a(10), b(:)
  allocatable :: a
  allocatable :: b
end subroutine

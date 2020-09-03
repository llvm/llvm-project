! RUN: %S/test_errors.sh %s %t %f18
subroutine s1
  integer :: x(2)
  !ERROR: The dimensions of 'x' have already been declared
  allocatable :: x(:)
  real :: y[1:*]
  !ERROR: The codimensions of 'y' have already been declared
  allocatable :: y[:]
end

subroutine s2
  target :: x(1)
  !ERROR: The dimensions of 'x' have already been declared
  integer :: x(2)
  target :: y[1:*]
  !ERROR: The codimensions of 'y' have already been declared
  integer :: y[2:*]
end

subroutine s3
  dimension :: x(4), x2(8)
  !ERROR: The dimensions of 'x' have already been declared
  allocatable :: x(:)
  codimension :: y[*], y2[1:2,2:*]
  !ERROR: The codimensions of 'y' have already been declared
  allocatable :: y[:]
end

subroutine s4
  integer, dimension(10) :: x(2,2), y
end

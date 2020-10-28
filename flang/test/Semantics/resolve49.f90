! RUN: %S/test_errors.sh %s %t %f18
! Test section subscript
program p1
  real :: a(10,10)
  real :: b(5,5)
  real :: c
  integer :: n
  n = 2
  b = a(1:10:n,1:n+3)
end

! Test substring
program p2
  type t1(n1,n2)
    integer,kind :: n1,n2
    integer :: c2(iachar('ABCDEFGHIJ'(n1:n1)))
  end type
  character :: a(10)
  character :: b(5)
  character :: c(0)
  integer :: n
  n = 3
  b = a(n:7)
  b = a(n+3:)
  b = a(:n+2)
  a(n:7) = b
  a(n+3:) = b
  a(:n+2) = b
  n = iachar(1_'ABCDEFGHIJ'(1:1))
  c = 'ABCDEFGHIJ'(1:0)
end

! Test pointer assignment with bounds
program p3
  integer, pointer :: a(:,:)
  integer, target :: b(2,2)
  integer :: n
  n = 2
  a(n:,n:) => b
  a(1:n,1:n) => b
end

! Test pointer assignment to array element
program p4
  type :: t
    real, pointer :: a
  end type
  type(t) :: x(10)
  integer :: i
  real, target :: y
  x(i)%a => y
end program

! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic
! A statement function dummy argument takes its type from the entity of the
! same name in the enclosing scope, which cannot be assumed-type.
subroutine s1(a)
  type(*) :: a
  integer :: func
  !ERROR: Statement function dummy argument 'a' cannot have assumed type
  func(a) = 1
end subroutine

! F2023 15.6.4
subroutine s2(a, n)
  integer :: a
  integer :: n
  character(len=n) :: func
  !PORTABILITY: The length of CHARACTER statement function result 'func' should be a constant expression [-Wstatement-function-extensions]
  func(a) = "hello"
end subroutine

subroutine s3(a)
  integer :: a
  integer :: c
  real :: func
  func(a) = 1
  c = func(a)
end subroutine

subroutine s4(a)
  integer :: a
  character(len=5) :: func
  func(a) = "hello"
end subroutine

subroutine s5(x)
  character(len=10) :: x
  character(len=10) :: func
  func(x) = x
end subroutine

subroutine s6(x, n)
  integer :: n
  character(len=n) :: x
  integer :: func
  !PORTABILITY: The length of CHARACTER statement function dummy argument 'x' should be a constant expression [-Wstatement-function-extensions]
  func(x) = 1
end subroutine

subroutine s7(a)
  type(*) :: a
  integer :: c
  integer :: func
  !ERROR: Statement function dummy argument 'a' cannot have assumed type
  func(a) = 1
  !ERROR: Assumed-type 'a' may be associated only with an assumed-type dummy argument 'a='
  c = func(a)
end subroutine

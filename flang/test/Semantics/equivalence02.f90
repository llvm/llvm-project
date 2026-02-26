! RUN: %python %S/test_errors.py %s %flang_fc1

program main
! ERROR: A sequence type should have at least one component [-Wempty-sequence-type]
type t0
  sequence
end type
type t1
  sequence
  integer, dimension(2):: i/41, 2/
end type
type t2
  sequence
  integer :: j(2) = [42, 2]
end type
type (t0) :: O1
! ERROR: Default component initializations of equivalenced objects affect 'a%i(1_8)' more than once, distinctly
type (t1) :: A
! ERROR: Default component initializations of equivalenced objects affect 'b%j(1_8)' more than once, distinctly
type (t2) :: B
! ERROR: Default component initializations of equivalenced objects affect 'x' more than once, distinctly
integer :: x, O2(0)
data x/42/
! ERROR: Default component initializations of equivalenced objects affect 'undeclared' more than once, distinctly
equivalence (A, B, x, O1, O2, Undeclared)
call p(x)
call s()
end

subroutine s()
  type g1
    sequence
    integer(kind=8)::d/1_8/
  end type
  type g2
    sequence
    integer(kind=8)::d = 2_8
  end type
  ! ERROR: Default component initializations of equivalenced objects affect 'c%d' more than once, distinctly
  type (g1) :: C
  ! ERROR: Default component initializations of equivalenced objects affect 'd%d' more than once, distinctly
  type (g2) :: D
  ! ERROR: Default component initializations of equivalenced objects affect 'x' more than once, distinctly
  ! ERROR: Default component initializations of equivalenced objects affect 'y' more than once, distinctly
  integer :: x, y
  data x/1/, y/2/
  equivalence (C, x)
  equivalence (D, y)
  equivalence (x, y)
  print *, x, y
end subroutine

subroutine p(x)
  integer :: x
  print *, x
end subroutine

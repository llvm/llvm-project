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
! ERROR: Distinct default component initializations of equivalenced objects affect 'o1' more than once
type (t0) :: O1
! ERROR: Distinct default component initializations of equivalenced objects affect 'a%i(1_8)' more than once
type (t1) :: A
! ERROR: Distinct default component initializations of equivalenced objects affect 'b%j(1_8)' more than once
type (t2) :: B
! ERROR: Distinct default component initializations of equivalenced objects affect 'x' more than once
! ERROR: Distinct default component initializations of equivalenced objects affect 'o2(1_8)' more than once
integer :: x, O2(0)
data x/42/
! ERROR: Distinct default component initializations of equivalenced objects affect 'undeclared' more than once
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
  ! ERROR: Distinct default component initializations of equivalenced objects affect 'c%d' more than once
  type (g1) :: C
  ! ERROR: Distinct default component initializations of equivalenced objects affect 'd%d' more than once
  type (g2) :: D
  ! ERROR: Distinct default component initializations of equivalenced objects affect 'x' more than once
  ! ERROR: Distinct default component initializations of equivalenced objects affect 'y' more than once
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

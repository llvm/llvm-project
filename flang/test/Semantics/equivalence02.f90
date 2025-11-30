! RUN: %python %S/test_errors.py %s %flang_fc1

program main
type t1
  sequence
  integer, dimension(2):: i/42, 1/      
end type
type t2
  sequence
  integer :: j(2) = [41, 1]
end type

! ERROR: Distinct default component initializations of equivalenced objects affect 'a%i(1_8)' more than once
type (t1) :: A
! ERROR: Distinct default component initializations of equivalenced objects affect 'b%j(1_8)' more than once
type (t2) :: B
! ERROR: Distinct default component initializations of equivalenced objects affect 'x' more than once
integer :: x
data x/42/
equivalence (A, B, x)
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

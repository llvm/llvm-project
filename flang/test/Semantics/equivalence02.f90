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
type pdt(k)
  integer, kind :: k
  ! NOTE: If you uncomment the sequence attribute, you get the following error:
  ! NOTE: A sequence type may not have type parameters
  !sequence
  ! NOTE: If you try to use a legacy style initialization, you get the following error:
  ! NOTE: Component 'x' in a parameterized data type may not be initialized with a legacy DATA-style value list
  integer, dimension(k + 1):: x = [43, (i, i=2, k+1, 1)]
end type
! ERROR: Distinct default component initializations of equivalenced objects affect 'o1' more than once
type (t0) :: O1
! ERROR: Distinct default component initializations of equivalenced objects affect 'a%i(1_8)' more than once
type (t1) :: A
! ERROR: Distinct default component initializations of equivalenced objects affect 'b%j(1_8)' more than once
type (t2) :: B
type (pdt(2)) :: P2
type (pdt(3)) :: P3
! ERROR: Distinct default component initializations of equivalenced objects affect 'x' more than once
! ERROR: Distinct default component initializations of equivalenced objects affect 'o2(1_8)' more than once
integer :: x, O2(0)
data x/42/
! NOTE: If you add P2 and P3 to the equivalence set, you get the following errors:
! NOTE: Nonsequence derived type object 'p2' is not allowed in an equivalence set
! NOTE: Nonsequence derived type object 'p3' is not allowed in an equivalence set
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

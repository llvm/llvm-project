! RUN: %python %S/test_errors.py %s %flang_fc1

! Make sure parameterized derived types can't be used in equivalence sets or contain legacy DATA-style initializations.
program main
type t1
  sequence
  integer, dimension(2):: i/41, 2/      
end type
type pdt(k)
  integer, kind :: k
  integer, dimension(k + 1):: x = [43, (i, i=2, k+1, 1)]
end type
type pdt1(k)
  integer, kind :: k
  !ERROR: Component 'x' in a parameterized data type may not be initialized with a legacy DATA-style value list
  integer, dimension(k + 1):: x/42, 2/
end type
!ERROR: A sequence type may not have type parameters
type pdt2(k)
  integer, kind :: k
  sequence
  integer, dimension(k + 1):: x = [43, (i, i=2, k+1, 1)]
end type

type (t1) :: A
type (pdt(2)) :: P2
type (pdt(3)) :: P3
type (pdt1(4)) :: P4
type (pdt2(5)) :: P5
! ERROR: Nonsequence derived type object 'p2' is not allowed in an equivalence set
! ERROR: Nonsequence derived type object 'p3' is not allowed in an equivalence set
! ERROR: Nonsequence derived type object 'p4' is not allowed in an equivalence set
equivalence (A, B, P2, P3, P4, P5)
end

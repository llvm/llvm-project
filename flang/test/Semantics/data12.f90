! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic
type :: t1
  sequence
  integer :: m = 123
  integer :: pad
end type
type :: t2
  sequence
  integer :: n = 123
  integer :: pad
end type
type :: t3
  sequence
  integer :: k = 234
  integer :: pad
end type
!ERROR: Default component initializations of equivalenced objects affect 'x1a%m' more than once, distinctly
type(t1) :: x1a
!ERROR: Default component initializations of equivalenced objects affect 'x2a%n' more than once, distinctly
type(t2) :: x2a
!ERROR: Default component initializations of equivalenced objects affect 'x3%k' more than once, distinctly
type(t3), save :: x3
!ERROR: Explicit initializations of equivalenced objects affect 'ja(2_8)' more than once, distinctly
!ERROR: Explicit initializations of equivalenced objects affect 'ka(1_8)' more than once, distinctly
integer :: ja(2), ka(2)
data ja/345, 456/
data ka/456, 567/
equivalence(x1a, x2a, x3)
! Same value: no error
type(t1) :: x1b
type(t2) :: x2b
equivalence(x1b, x2b)
equivalence(ja(2),ka(1))
end

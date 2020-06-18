! RUN: %S/test_symbols.sh %s %t %f18 -fopenmp

! Test clauses that accept list.
! 2.1 Directive Format
!   A list consists of a comma-separated collection of one or more list items.
!   A list item is a variable, array section or common block name (enclosed in
!   slashes).

!DEF: /md Module
module md
 !DEF: /md/myty PUBLIC DerivedType
 type :: myty
  !DEF: /md/myty/a ObjectEntity REAL(4)
  real :: a
  !DEF: /md/myty/b ObjectEntity INTEGER(4)
  integer :: b
 end type myty
end module md
!DEF: /mm MainProgram
program mm
 !REF: /md
 use :: md
 !DEF: /mm/c CommonBlockDetails
 !DEF: /mm/x ObjectEntity REAL(4)
 !DEF: /mm/y ObjectEntity REAL(4)
 common /c/x, y
 !REF: /mm/x
 !REF: /mm/y
 real x, y
 !DEF: /mm/myty Use
 !DEF: /mm/t ObjectEntity TYPE(myty)
 type(myty) :: t
 !DEF: /mm/b ObjectEntity INTEGER(4)
 integer b(10)
 !REF: /mm/t
 !REF: /md/myty/a
 t%a = 3.14
 !REF: /mm/t
 !REF: /md/myty/b
 t%b = 1
 !REF: /mm/b
 b = 2
 !DEF: /mm/a (Implicit) ObjectEntity REAL(4)
 a = 1.0
 !DEF: /mm/c (Implicit) ObjectEntity REAL(4)
 c = 2.0
!$omp parallel do  private(a,t,/c/) shared(c)
 !DEF: /mm/Block1/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
 do i=1,10
  !DEF: /mm/Block1/a (OmpPrivate) HostAssoc REAL(4)
  !REF: /mm/b
  !REF: /mm/Block1/i
  a = a+b(i)
  !DEF: /mm/Block1/t (OmpPrivate) HostAssoc TYPE(myty)
  !REF: /md/myty/a
  !REF: /mm/Block1/i
  t%a = i
  !DEF: /mm/Block1/y (OmpPrivate) HostAssoc REAL(4)
  y = 0.
  !DEF: /mm/Block1/x (OmpPrivate) HostAssoc REAL(4)
  !REF: /mm/Block1/a
  !REF: /mm/Block1/i
  !REF: /mm/Block1/y
  x = a+i+y
  !REF: /mm/c
  c = 3.0
 end do
end program

! RUN: %python %S/test_symbols.py %s %flang_fc1
! Allow redeclaration of inherited inaccessible components
!DEF: /m1 Module
module m1
 !DEF: /m1/t0 PRIVATE DerivedType
 type, private :: t0
 end type
 !REF: /m1/t0
 !DEF: /m1/t1 PUBLIC DerivedType
 type, extends(t0) :: t1
  !DEF: /m1/t1/n1a PRIVATE ObjectEntity INTEGER(4)
  !DEF: /m1/t1/n1b PRIVATE ObjectEntity INTEGER(4)
  integer, private :: n1a = 1, n1b = 2
 end type
end module
!DEF: /m2 Module
module m2
 !REF: /m1
 use :: m1
 !DEF: /m2/t1 PUBLIC Use
 !DEF: /m2/t2 PUBLIC DerivedType
 type, extends(t1) :: t2
  !DEF: /m2/t2/t0 ObjectEntity REAL(4)
  real :: t0
  !DEF: /m2/t2/n1a ObjectEntity REAL(4)
  real :: n1a
 end type
 !REF: /m2/t2
 !DEF: /m2/t3 PUBLIC DerivedType
 type, extends(t2) :: t3
  !DEF: /m2/t3/n1b ObjectEntity REAL(4)
  real :: n1b
 end type
end module
!DEF: /test (Subroutine) Subprogram
subroutine test
 !REF: /m2
 use :: m2
 !DEF: /test/t3 Use
 !DEF: /test/x ObjectEntity TYPE(t3)
 type(t3) :: x
 !REF: /test/x
 !REF: /m2/t3/n1b
 x%n1b = 1.
 !REF: /test/x
 !DEF: /m2/t3/t2 (ParentComp) ObjectEntity TYPE(t2)
 !DEF: /test/t2 Use
 x%t2 = t2(t0=2., n1a=3.)
 !REF: /test/x
 !REF: /m2/t2/t0
 x%t0 = 4.
 !REF: /test/x
 !REF: /m2/t2/n1a
 x%n1a = 5.
end subroutine

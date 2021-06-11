! RUN: %S/test_symbols.sh %s %t %flang_fc1
! REQUIRES: shell
!DEF: /s1 (Subroutine) Subprogram
subroutine s1
 implicit none
 !DEF: /s1/x ObjectEntity REAL(8)
 real(kind=8) :: x = 2.0
 !DEF: /s1/a ObjectEntity INTEGER(4)
 integer a
 !DEF: /s1/t DerivedType
 type :: t
 end type
 !REF: /s1/t
 !DEF: /s1/z ALLOCATABLE ObjectEntity CLASS(t)
 class(t), allocatable :: z
 !DEF: /s1/Block1/a AssocEntity REAL(8)
 !REF: /s1/x
 !DEF: /s1/Block1/b AssocEntity REAL(8)
 !DEF: /s1/Block1/c AssocEntity CLASS(t)
 !REF: /s1/z
 associate (a => x, b => x+1, c => z)
  !REF: /s1/x
  !REF: /s1/Block1/a
  x = a
 end associate
end subroutine

!DEF: /s2 (Subroutine) Subprogram
subroutine s2
 !DEF: /s2/x ObjectEntity CHARACTER(4_4,1)
 !DEF: /s2/y ObjectEntity CHARACTER(4_4,1)
 character(len=4) x, y
 !DEF: /s2/Block1/z AssocEntity CHARACTER(4_8,1)
 !REF: /s2/x
 associate (z => x)
  !REF: /s2/Block1/z
  print *, "z:", z
 end associate
 !TODO: need correct length for z
 !DEF: /s2/Block2/z AssocEntity CHARACTER(8_8,1)
 !REF: /s2/x
 !REF: /s2/y
 associate (z => x//y)
  !REF: /s2/Block2/z
  print *, "z:", z
 end associate
end subroutine

!DEF: /s3 (Subroutine) Subprogram
subroutine s3
 !DEF: /s3/t1 DerivedType
 type :: t1
  !DEF: /s3/t1/a1 ObjectEntity INTEGER(4)
  integer :: a1
 end type
 !REF: /s3/t1
 !DEF: /s3/t2 DerivedType
 type, extends(t1) :: t2
  !DEF: /s3/t2/a2 ObjectEntity INTEGER(4)
  integer :: a2
 end type
 !DEF: /s3/i ObjectEntity INTEGER(4)
 integer i
 !REF: /s3/t1
 !DEF: /s3/x POINTER ObjectEntity CLASS(t1)
 class(t1), pointer :: x
 !REF: /s3/x
 select type (y => x)
  !REF: /s3/t2
  class is (t2)
   !REF: /s3/i
   !DEF: /s3/Block1/y TARGET AssocEntity TYPE(t2)
   !REF: /s3/t2/a2
   i = y%a2
  !REF: /s3/t1
  type is (t1)
   !REF: /s3/i
   !DEF: /s3/Block2/y TARGET AssocEntity TYPE(t1)
   !REF: /s3/t1/a1
   i = y%a1
  class default
   !DEF: /s3/Block3/y TARGET AssocEntity CLASS(t1)
   print *, y
 end select
end subroutine

!DEF: /s4 (Subroutine) Subprogram
subroutine s4
 !DEF: /s4/t1 DerivedType
 type :: t1
  !DEF: /s4/t1/a ObjectEntity REAL(4)
  real :: a
 end type
 !DEF: /s4/t2 DerivedType
 type :: t2
  !REF: /s4/t1
  !DEF: /s4/t2/b ObjectEntity TYPE(t1)
  type(t1) :: b
 end type
 !REF: /s4/t2
 !DEF: /s4/x ObjectEntity TYPE(t2)
 type(t2) :: x
 !DEF: /s4/Block1/y AssocEntity TYPE(t1)
 !REF: /s4/x
 !REF: /s4/t2/b
 associate(y => x%b)
  !REF: /s4/Block1/y
  !REF: /s4/t1/a
  y%a = 0.0
 end associate
end subroutine

!DEF: /s5 (Subroutine) Subprogram
subroutine s5
 !DEF: /s5/t DerivedType
 type :: t
  !DEF: /s5/t/a ObjectEntity REAL(4)
  real :: a
 end type
 !DEF: /s5/b ObjectEntity REAL(4)
 real b
 !DEF: /s5/Block1/x AssocEntity TYPE(t)
 !DEF: /s5/f (Function) Subprogram TYPE(t)
 associate(x => f())
  !REF: /s5/b
  !REF: /s5/Block1/x
  !REF: /s5/t/a
  b = x%a
 end associate
contains
 !REF: /s5/f
 function f()
  !REF: /s5/t
  !DEF: /s5/f/f ObjectEntity TYPE(t)
  type(t) :: f
 end function
end subroutine

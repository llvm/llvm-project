! RUN: %python %S/test_symbols.py %s %flang_fc1
! Derived type forward reference regression case

 !DEF: /MainProgram1/t2 DerivedType
 type :: t2
  !DEF: /MainProgram1/t1 DerivedType
  !DEF: /MainProgram1/t2/ptr POINTER ObjectEntity TYPE(t1)
  type(t1), pointer :: ptr
 end type
 !REF: /MainProgram1/t1
 type :: t1
  !DEF: /MainProgram1/t1/a ObjectEntity REAL(4)
  real :: a
  !REF: /MainProgram1/t2
  !DEF: /MainProgram1/t1/p2 POINTER ObjectEntity TYPE(t2)
  type(t2), pointer :: p2
  !REF: /MainProgram1/t1
  !DEF: /MainProgram1/t1/p1 POINTER ObjectEntity TYPE(t1)
  type(t1), pointer :: p1
 end type
 !REF: /MainProgram1/t1
 !DEF: /MainProgram1/x1 POINTER ObjectEntity TYPE(t1)
 !DEF: /MainProgram1/x2 POINTER ObjectEntity TYPE(t1)
 type(t1), pointer :: x1, x2
 !REF: /MainProgram1/x2
 !REF: /MainProgram1/t1/p1
 !REF: /MainProgram1/t1/a
 !REF: /MainProgram1/x1
 x2%p1%a = x1%a
end program

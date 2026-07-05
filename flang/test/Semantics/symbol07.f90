! RUN: %python %S/test_symbols.py %s %flang_fc1
!DEF: /MAIN MainProgram
program MAIN
 implicit complex(z)
 !DEF: /MAIN/t DerivedType
 type :: t
  !DEF: /MAIN/t/re ObjectEntity REAL(4)
  real :: re
  !DEF: /MAIN/t/im ObjectEntity REAL(4)
  real :: im
 end type
 !DEF: /MAIN/z1 ObjectEntity COMPLEX(4)
 complex z1
 !REF: /MAIN/t
 !DEF: /MAIN/w ObjectEntity TYPE(t)
 type(t) :: w
 !DEF: /MAIN/x ObjectEntity REAL(4)
 !DEF: /MAIN/y ObjectEntity REAL(4)
 real x, y
 !REF: /MAIN/x
 !REF: /MAIN/z1
 x = z1%re
 !REF: /MAIN/y
 !REF: /MAIN/z1
 y = z1%im
 !DEF: /MAIN/z2 (Implicit) ObjectEntity COMPLEX(4)
 !REF: /MAIN/x
 z2%re = x
 !REF: /MAIN/z2
 !REF: /MAIN/y
 z2%im = y
 !REF: /MAIN/x
 !REF: /MAIN/w
 !REF: /MAIN/t/re
 x = w%re
 !REF: /MAIN/y
 !REF: /MAIN/w
 !REF: /MAIN/t/im
 y = w%im
end program

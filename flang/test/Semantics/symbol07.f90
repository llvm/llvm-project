! RUN: %S/test_symbols.sh %s %t %f18
!DEF: /main MainProgram
program main
 implicit complex(z)
 !DEF: /main/t DerivedType
 type :: t
  !DEF: /main/t/re ObjectEntity REAL(4)
  real :: re
  !DEF: /main/t/im ObjectEntity REAL(4)
  real :: im
 end type
 !DEF: /main/z1 ObjectEntity COMPLEX(4)
 complex z1
 !REF: /main/t
 !DEF: /main/w ObjectEntity TYPE(t)
 type(t) :: w
 !DEF: /main/x ObjectEntity REAL(4)
 !DEF: /main/y ObjectEntity REAL(4)
 real x, y
 !REF: /main/x
 !REF: /main/z1
 x = z1%re
 !REF: /main/y
 !REF: /main/z1
 y = z1%im
 !DEF: /main/z2 (Implicit) ObjectEntity COMPLEX(4)
 !REF: /main/x
 z2%re = x
 !REF: /main/z2
 !REF: /main/y
 z2%im = y
 !REF: /main/x
 !REF: /main/w
 !REF: /main/t/re
 x = w%re
 !REF: /main/y
 !REF: /main/w
 !REF: /main/t/im
 y = w%im
end program

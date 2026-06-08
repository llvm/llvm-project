! RUN: %python %S/test_symbols.py %s %flang_fc1 -fopenacc
!DEF: /MAIN MainProgram
program MAIN
 !DEF: /MAIN/t ABSTRACT DerivedType
 type, abstract :: t
 end type
 !REF: /MAIN/t
 !DEF: /MAIN/t2 DerivedType
 type, extends(t) :: t2
  !DEF: /MAIN/t2/y ObjectEntity REAL(4)
  real :: y
 end type
contains
 !DEF: /MAIN/s (Subroutine) Subprogram
 !DEF: /MAIN/s/d ObjectEntity CLASS(t)
 subroutine s (d)
  !REF: /MAIN/t
  !REF: /MAIN/s/d
  class(t) :: d
  !DEF: /MAIN/s/a ObjectEntity REAL(4)
  real a
!$acc data create(a)
  !REF: /MAIN/s/d
  select type (d)
   !REF: /MAIN/t2
  class is (t2)
   !DEF: /MAIN/s/OpenACCConstruct1/OtherConstruct1/d AssocEntity CLASS(t2)
   !REF: /MAIN/t2/y
   d%y = 1.
  end select
!$acc end data
 end subroutine
end program

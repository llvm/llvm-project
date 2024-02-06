! RUN: %python %S/test_symbols.py %s %flang_fc1
! Test the executable part skimming for apparent calls, to ensure that
! symbols in nested scopes (BLOCK, &c.) properly shadow host symbols.
!DEF: /m Module
module m
end module
!DEF: /subr (Subroutine) Subprogram
!DEF: /subr/da INTENT(IN) ObjectEntity CLASS(*)
!DEF: /subr/ar INTENT(IN) ObjectEntity REAL(4)
subroutine subr (da, ar)
 !REF: /subr/da
 class(*), intent(in) :: da(:)
 !REF: /subr/ar
 real, intent(in) :: ar(..)
 !DEF: /subr/s2 ObjectEntity REAL(4)
 !DEF: /subr/s4 ObjectEntity REAL(4)
 !DEF: /subr/s6 ObjectEntity REAL(4)
 !DEF: /subr/s7 (Function) ProcEntity REAL(4)
 !DEF: /subr/s8 ObjectEntity REAL(4)
 real s2, s4, s6, s7, s8
 !DEF: /s1 EXTERNAL (Function, Implicit) ProcEntity REAL(4)
 print *, s1(1)
 block
  !DEF: /subr/BlockConstruct1/s2 ObjectEntity INTEGER(4)
  !DEF: /subr/BlockConstruct1/s5 (Function) ProcEntity INTEGER(4)
  integer s2(10), s5
  !DEF: /subr/BlockConstruct1/s4 DerivedType
  type :: s4
   !DEF: /subr/BlockConstruct1/s4/n ObjectEntity INTEGER(4)
   integer :: n
  end type
  !REF: /subr/BlockConstruct1/s2
  print *, s2(1)
  !DEF: /s3 EXTERNAL (Function, Implicit) ProcEntity REAL(4)
  print *, s3(1)
  !REF: /subr/BlockConstruct1/s4
  print *, s4(1)
  !REF: /subr/BlockConstruct1/s5
  print *, s5(1)
 end block
 block
  import, none
  !DEF: /s2 EXTERNAL (Function, Implicit) ProcEntity REAL(4)
  print *, s2(1)
 end block
 block
  !REF: /subr/s6
  import, only: s6
  !DEF: /s8 EXTERNAL (Function, Implicit) ProcEntity REAL(4)
  print *, s8(1)
 end block
 block
  !REF: /m
  use :: m
  !REF: /subr/s7
  print *, s7(1)
 end block
 !DEF: /subr/OtherConstruct1/s2 AssocEntity REAL(4)
 associate (s2 => [1.])
  !REF: /subr/OtherConstruct1/s2
  print *, s2(1)
 end associate
 !REF: /subr/da
 select type (s2 => da)
 type is (real)
  !DEF: /subr/OtherConstruct2/s2 AssocEntity REAL(4)
  print *, s2(1)
 end select
 !REF: /subr/ar
 select rank (s2 => ar)
 rank (1)
  !DEF: /subr/OtherConstruct3/s2 AssocEntity REAL(4)
  print *, s2(1)
 end select
end subroutine

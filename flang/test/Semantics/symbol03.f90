! RUN: %python %S/test_symbols.py %s %flang_fc1
! Test host association in internal subroutine of main program.

!DEF: /main MainProgram
program main
 !DEF: /main/x ObjectEntity INTEGER(4)
 integer x
 !DEF: /main/s (Subroutine) Subprogram
 call s
contains
 !REF: /main/s
 subroutine s
  !DEF: /main/s/y (Implicit) ObjectEntity REAL(4)
  !DEF: /main/s/x HostAssoc INTEGER(4)
  y = x
 contains
  !DEF: /main/s/s2 (Subroutine) Subprogram
  subroutine s2
   !DEF: /main/s/s2/z (Implicit) ObjectEntity REAL(4)
   !DEF: /main/s/s2/x HostAssoc INTEGER(4)
   z = x
  end subroutine
 end subroutine
end program

!DEF: /s (Subroutine) Subprogram
subroutine s
 !DEF: /s/x ObjectEntity REAL(4)
 real x(100, 100)
 !DEF: /s/s1 (Subroutine) Subprogram
 call s1
contains
 !REF: /s/s1
  subroutine s1
    !DEF: /s/s1/x HostAssoc REAL(4)
    print *, x(10, 10)
  end subroutine
end subroutine

!DEF: /sb (Subroutine) Subprogram
subroutine sb
 !DEF: /sb/x TARGET ObjectEntity REAL(4)
 real, target :: x
 !DEF: /sb/s1 (Subroutine) Subprogram
 call s1
contains
 !REF: /sb/s1
 subroutine s1
  !DEF: /sb/s1/p POINTER ObjectEntity REAL(4)
  real, pointer :: p
  !REF: /sb/s1/p
  !DEF: /sb/s1/x TARGET HostAssoc REAL(4)
  p => x
 end subroutine
end subroutine

! Test host associated symbols are also created for symbols that are use
! associated in the host.

!DEF: /m1 Module
module m1
 !DEF: /m1/x PUBLIC ObjectEntity REAL(4)
 real x(100,100)
 !DEF: /m1/x_target PUBLIC, TARGET ObjectEntity REAL(4)
 real, target :: x_target
end module

!DEF: /s_use (Subroutine) Subprogram
subroutine s_use
 !REF: /m1
 use :: m1
 !DEF: /s_use/x Use REAL(4)
 print *, x
 !DEF: /s_use/s1 (Subroutine) Subprogram
 call s1
contains
 !REF: /s_use/s1
 subroutine s1
  !DEF: /s_use/s1/x HostAssoc REAL(4)
  print *, x(10,10)
 end subroutine
end subroutine

!DEF: /sb_use (Subroutine) Subprogram
subroutine sb_use
 !REF: /m1
 use :: m1
 !DEF: /sb_use/x_target TARGET Use REAL(4)
 print *, x_target
 !DEF: /sb_use/s1 (Subroutine) Subprogram
 call s1
contains
 !REF: /sb_use/s1
 subroutine s1
  !DEF: /sb_use/s1/p POINTER ObjectEntity REAL(4)
  real, pointer :: p
  !REF: /sb_use/s1/p
  !DEF: /sb_use/s1/x_target TARGET HostAssoc REAL(4)
  p => x_target
 end subroutine
end subroutine

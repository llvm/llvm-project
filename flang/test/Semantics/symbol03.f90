! RUN: %S/test_symbols.sh %s %t %f18
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

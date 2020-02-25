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
  !REF: /main/x
  y = x
 end subroutine
end program

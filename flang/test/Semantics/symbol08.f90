! RUN: %python %S/test_symbols.py %s %flang_fc1
!DEF: /MAIN MainProgram
program MAIN
 !DEF: /MAIN/x POINTER ObjectEntity REAL(4)
 pointer :: x
 !REF: /MAIN/x
 real x
 !DEF: /MAIN/y EXTERNAL, POINTER (Function) ProcEntity REAL(4)
 pointer :: y
 !REF: /MAIN/y
 procedure (real) :: y
 !DEF: /MAIN/z (Implicit) ObjectEntity REAL(4)
 !REF: /MAIN/y
 z = y()
end program

! RUN: %S/test_symbols.sh %s %t %f18
!DEF: /main MainProgram
program main
 !DEF: /main/x POINTER ObjectEntity REAL(4)
 pointer :: x
 !REF: /main/x
 real x
 !DEF: /main/y EXTERNAL, POINTER (Function) ProcEntity REAL(4)
 pointer :: y
 !REF: /main/y
 procedure (real) :: y
 !DEF: /main/z (Implicit) ObjectEntity REAL(4)
 !REF: /main/y
 z = y()
end program

! RUN: %python %S/test_symbols.py %s %flang_fc1
! Statement functions

!DEF: /P1 MainProgram
program P1
 !DEF: /P1/f (Function, StmtFunction) Subprogram INTEGER(4)
 !DEF: /P1/i ObjectEntity INTEGER(4)
 !DEF: /P1/j ObjectEntity INTEGER(4)
 integer f, i, j
 !REF: /P1/f
 !REF: /P1/i
 !DEF: /P1/f/i ObjectEntity INTEGER(4)
 f(i) = i + 1
 !REF: /P1/j
 !REF: /P1/f
 j = f(2)
end program

!DEF: /p2 (Subroutine)Subprogram
subroutine p2
 !DEF: /p2/f (Function, StmtFunction) Subprogram REAL(4)
 !DEF: /p2/f/x (Implicit) ObjectEntity REAL(4)
 !DEF: /p2/y (Implicit) ObjectEntity REAL(4)
 f(x) = y
 !REF: /p2/y
 y = 1.0
end subroutine

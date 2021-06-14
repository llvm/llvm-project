! RUN: %S/test_symbols.sh %s %t %flang_fc1
! REQUIRES: shell
! Statement functions

!DEF: /p1 MainProgram
program p1
 !DEF: /p1/f (Function, StmtFunction) Subprogram INTEGER(4)
 !DEF: /p1/i ObjectEntity INTEGER(4)
 !DEF: /p1/j ObjectEntity INTEGER(4)
 integer f, i, j
 !REF: /p1/f
 !REF: /p1/i
 !DEF: /p1/f/i ObjectEntity INTEGER(4)
 f(i) = i + 1
 !REF: /p1/j
 !REF: /p1/f
 j = f(2)
end program

!DEF: /p2 MainProgram
program p2
 !DEF: /p2/f (Function, StmtFunction) Subprogram REAL(4)
 !DEF: /p2/f/x (Implicit) ObjectEntity REAL(4)
 !DEF: /p2/y (Implicit) ObjectEntity REAL(4)
 f(x) = y
 !REF: /p2/y
 y = 1.0
end program

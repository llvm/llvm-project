! Statement functions

!DEF: /p1 MainProgram
program p1
 !DEF: /p1/f (Function) Subprogram INTEGER(4)
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

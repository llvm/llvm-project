! RUN: %S/test_symbols.sh %s %t %f18

! Intrinsic function in type declaration statement: type is ignored

!DEF: /p1 MainProgram
program p1
 !DEF: /p1/cos INTRINSIC (Function) ProcEntity
 integer cos
 !DEF: /p1/y (Implicit) ObjectEntity REAL(4)
 !REF: /p1/cos
 !DEF: /p1/x (Implicit) ObjectEntity REAL(4)
 y = cos(x)
 !REF: /p1/y
 !DEF: /p1/sin INTRINSIC (Function) ProcEntity
 !REF: /p1/x
 y = sin(x)
 !REF: /p1/y
 !DEF: /f EXTERNAL (Function, Implicit) ProcEntity REAL(4)
 !REF: /p1/x
 y = f(x)
end program

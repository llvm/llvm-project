! RUN: %python %S/test_symbols.py %s %flang_fc1

! Intrinsic function in type declaration statement: type is ignored

!DEF: /p1 MainProgram
program p1
 !DEF: /p1/cos ELEMENTAL, INTRINSIC, PURE (Function) ProcEntity INTEGER(4)
 integer cos
 !DEF: /p1/y (Implicit) ObjectEntity REAL(4)
 !REF: /p1/cos
 !DEF: /p1/x (Implicit) ObjectEntity REAL(4)
 y = cos(x)
 !REF: /p1/y
 !DEF: /p1/sin ELEMENTAL, INTRINSIC, PURE (Function) ProcEntity
 !REF: /p1/x
 y = sin(x)
 !REF: /p1/y
 !DEF: /f EXTERNAL (Function, Implicit) ProcEntity REAL(4)
 !REF: /p1/x
 y = f(x)
end program

!DEF: /f2 (Function) Subprogram REAL(4)
!DEF: /f2/cos EXTERNAL (Function, Implicit) ProcEntity REAL(4)
!DEF: /f2/x (Implicit) ObjectEntity REAL(4)
function f2(cos, x)
 !DEF: /f2/f2 (Implicit) ObjectEntity REAL(4)
 !REF: /f2/cos
 !REF: /f2/x
 f2 = cos(x)
end function

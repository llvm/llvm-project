! RUN: %python %S/test_symbols.py %s %flang_fc1

! Intrinsic function in type declaration statement: type is ignored

!DEF: /P1 MainProgram
program P1
 !DEF: /P1/cos ELEMENTAL, INTRINSIC, PURE (Function) ProcEntity INTEGER(4)
 integer cos
 !DEF: /P1/y (Implicit) ObjectEntity REAL(4)
 !REF: /P1/cos
 !DEF: /P1/x (Implicit) ObjectEntity REAL(4)
 y = cos(x)
 !REF: /P1/y
 !DEF: /P1/sin ELEMENTAL, INTRINSIC, PURE (Function) ProcEntity
 !REF: /P1/x
 y = sin(x)
 !REF: /P1/y
 !DEF: /f EXTERNAL (Function, Implicit) ProcEntity REAL(4)
 !REF: /P1/x
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

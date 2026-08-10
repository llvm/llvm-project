! RUN: %python %S/test_symbols.py %s %flang_fc1
!DEF: /mod Module
module mod
contains
    !DEF: /mod/internal_definition_sin_real4 PUBLIC (Function) Subprogram REAL(4)
    !DEF: /mod/internal_definition_sin_real4/r4 ObjectEntity REAL(4)
    function internal_definition_sin_real4(r4)
        !REF: /mod/internal_definition_sin_real4/r4
        real r4
        !DEF: /mod/internal_definition_sin_real4/internal_definition_sin_real4 ObjectEntity REAL(4)
        real internal_definition_sin_real4
        !REF: /mod/internal_definition_sin_real4/internal_definition_sin_real4
        !REF: /mod/internal_definition_sin_real4/r4
        internal_definition_sin_real4 = r4+100
    end function
end module
!DEF: /MAIN MainProgram
program MAIN
    !REF: /mod
    use :: mod
    !DEF: /MAIN/sin ELEMENTAL, INTRINSIC, PURE (Function) Generic
    intrinsic :: sin
    !REF: /MAIN/sin
    interface sin
        !DEF: /MAIN/internal_definition_sin_real4 (Function) Use REAL(4)
        procedure :: internal_definition_sin_real4
    end interface
    !DEF: /MAIN/a ObjectEntity REAL(4)
    real a
    !REF: /MAIN/a
    !REF: /MAIN/internal_definition_sin_real4
    a = sin(1.0)
    !REF: /MAIN/a
    print *, a
end program

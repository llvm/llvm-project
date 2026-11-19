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

!DEF: /mod1 Module
module mod1
contains
    !DEF: /mod1/nonelemental_function_beats_intrinsic PUBLIC (Subroutine) Subprogram
    subroutine nonelemental_function_beats_intrinsic
        !REF: /mod
        use :: mod
        !DEF: /mod1/nonelemental_function_beats_intrinsic/sin ELEMENTAL, INTRINSIC, PURE (Function) Generic
        intrinsic :: sin
        !REF: /mod1/nonelemental_function_beats_intrinsic/sin
        interface sin
            !DEF: /mod1/nonelemental_function_beats_intrinsic/internal_definition_sin_real4 (Function) Use REAL(4)
            procedure :: internal_definition_sin_real4
        end interface
        !DEF: /mod1/nonelemental_function_beats_intrinsic/a ObjectEntity REAL(4)
        real a
        !DEF: /mod1/nonelemental_function_beats_intrinsic/b ObjectEntity REAL(8)
        real(kind=8) b
        !REF: /mod1/nonelemental_function_beats_intrinsic/a
        !REF: /mod1/nonelemental_function_beats_intrinsic/internal_definition_sin_real4
        a = sin(1.0)
        !REF: /mod1/nonelemental_function_beats_intrinsic/a
        print *, a
        !REF: /mod1/nonelemental_function_beats_intrinsic/b
        !REF: /mod1/nonelemental_function_beats_intrinsic/sin
        b = sin(1.0_8)
    end subroutine
end module

!DEF: /mod2 Module
module mod2
contains
    !DEF: /mod2/elemental_specific_sin_real4 ELEMENTAL, PUBLIC (Function) Subprogram REAL(4)
    !DEF: /mod2/elemental_specific_sin_real4/x INTENT(IN) ObjectEntity REAL(4)
    elemental real function elemental_specific_sin_real4(x)
        !REF: /mod2/elemental_specific_sin_real4/x
        real, intent(in) :: x
        !DEF: /mod2/elemental_specific_sin_real4/elemental_specific_sin_real4 ObjectEntity REAL(4)
        !REF: /mod2/elemental_specific_sin_real4/x
        elemental_specific_sin_real4 = x + 100.
    end function
    !DEF: /mod2/elemental_specific_beats_intrinsic PUBLIC (Subroutine) Subprogram
    subroutine elemental_specific_beats_intrinsic
        !DEF: /mod2/elemental_specific_beats_intrinsic/sin ELEMENTAL, INTRINSIC, PURE (Function) Generic
        intrinsic :: sin
        !REF: /mod2/elemental_specific_beats_intrinsic/sin
        interface sin
            !REF: /mod2/elemental_specific_sin_real4
            procedure :: elemental_specific_sin_real4
        end interface
        !DEF: /mod2/elemental_specific_beats_intrinsic/a ObjectEntity REAL(4)
        real a
        !REF: /mod2/elemental_specific_beats_intrinsic/a
        !REF: /mod2/elemental_specific_sin_real4
        a = sin(1.0)
        !REF: /mod2/elemental_specific_beats_intrinsic/a
        print *, a
    end subroutine
end module

!DEF: /mod3 Module
module mod3
contains
    !DEF: /mod3/subroutine_specific_cpu_time PUBLIC (Subroutine) Subprogram
    !DEF: /mod3/subroutine_specific_cpu_time/x INTENT(OUT) ObjectEntity REAL(4)
    subroutine subroutine_specific_cpu_time(x)
        !REF: /mod3/subroutine_specific_cpu_time/x
        real, intent(out) :: x
        !REF: /mod3/subroutine_specific_cpu_time/x
        x = 100.
    end subroutine
    !DEF: /mod3/subroutine_generic_beats_intrinsic PUBLIC (Subroutine) Subprogram
    subroutine subroutine_generic_beats_intrinsic
        !DEF: /mod3/subroutine_generic_beats_intrinsic/cpu_time INTRINSIC (Subroutine) Generic
        intrinsic :: cpu_time
        !REF: /mod3/subroutine_generic_beats_intrinsic/cpu_time
        interface cpu_time
            !REF: /mod3/subroutine_specific_cpu_time
            procedure :: subroutine_specific_cpu_time
        end interface
        !DEF: /mod3/subroutine_generic_beats_intrinsic/t ObjectEntity REAL(4)
        real t
        !REF: /mod3/subroutine_specific_cpu_time
        !REF: /mod3/subroutine_generic_beats_intrinsic/t
        call cpu_time(t)
        !REF: /mod3/subroutine_generic_beats_intrinsic/t
        print *, t
    end subroutine
end module

!DEF: /m Module
module m
    !DEF: /m/sin ELEMENTAL, INTRINSIC, PUBLIC, PURE (Function) Generic
    intrinsic :: sin
    !REF: /m/sin
    interface sin
        !DEF: /m/fr4 PUBLIC (Function) Subprogram REAL(4)
        procedure :: fr4
    end interface
contains
    !REF: /m/fr4
    !DEF: /m/fr4/r4 ObjectEntity REAL(4)
    function fr4(r4)
        !REF: /m/fr4/r4
        real r4
        !DEF: /m/fr4/fr4 (Implicit) ObjectEntity REAL(4)
        !REF: /m/fr4/r4
        fr4 = r4 + 100
    end function
end module
!DEF: /mod4 Module
module mod4
contains
    !DEF: /mod4/use_associated_generic_and_intrinsic PUBLIC (Subroutine) Subprogram
    subroutine use_associated_generic_and_intrinsic
        !REF: /m
        use :: m
        !DEF: /mod4/use_associated_generic_and_intrinsic/a ObjectEntity REAL(4)
        real a
        !DEF: /mod4/use_associated_generic_and_intrinsic/b ObjectEntity REAL(8)
        real(kind=8) b
        !REF: /mod4/use_associated_generic_and_intrinsic/a
        !REF: /m/fr4
        a = sin(0.1)
        !REF: /mod4/use_associated_generic_and_intrinsic/a
        print *, a
        !REF: /mod4/use_associated_generic_and_intrinsic/b
        !REF: /m/sin
        b = sin(0.1_8)
        !REF: /mod4/use_associated_generic_and_intrinsic/b
        print *, b
    end subroutine
end module

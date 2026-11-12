! RUN: %python %S/test_errors.py %s %flang_fc1
!ERROR: Symbol has a BIND(C) name that is not a valid C language identifier
subroutine bang() bind(C,name='!')
end
!ERROR: Symbol has a BIND(C) name that is not a valid C language identifier
subroutine cr() bind(C,name=achar(13))
end
!ERROR: Symbol has a BIND(C) name that is not a valid C language identifier
subroutine beast() bind(C,name="666")
end

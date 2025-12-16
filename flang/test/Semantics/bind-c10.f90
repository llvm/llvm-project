! RUN: %python %S/test_errors.py %s %flang_fc1
subroutine bang() bind(C,name='!')
end
!ERROR: Symbol has a BIND(C) name containing non-visible ASCII character(s)
subroutine cr() bind(C,name=achar(13))
end
subroutine beast() bind(C,name="666")
end
!ERROR: Symbol has a BIND(C) name with leading dot that may have special meaning to the toolchain
subroutine llvm() bind(C,name=".L1")
end
!ERROR: Symbol has a BIND(C) name with leading dot that may have special meaning to the toolchain
subroutine ld() bind(C,name=".text")
end

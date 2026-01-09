! RUN: %python %S/test_errors.py %s %flang_fc1
subroutine bang() bind(C,name='!')
end
!ERROR: Symbol has a BIND(C) name containing non-visible ASCII character(s)
subroutine cr() bind(C,name=achar(13))
end
!! depending on used assembler it can be threated as error or not
subroutine beast() bind(C,name="666")
end
!! depending on used assembler it can be threated as error or not
subroutine llvm() bind(C,name=".L1")
end
!! depending on used assembler it can be threated as error or not
subroutine as() bind(C,name=".text")
end
!! depending on used assembler it can be threated as error or not
subroutine printable_ascii() bind(C,name="! ""#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~'")
end

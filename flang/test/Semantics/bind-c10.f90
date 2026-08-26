! RUN: %python %S/test_errors.py %s %flang_fc1
subroutine foo() bind(C,name='bar')
end
subroutine currency() bind(C,name='$')
end
!ERROR: Symbol has a BIND(C) name containing non-visible ASCII character(s)
subroutine cr() bind(C,name=achar(13))
end
!ERROR: Symbol has a BIND(C) name containing non-visible ASCII character(s)
subroutine null_terminator() bind(C,name="null_terminator" // achar(0))
end

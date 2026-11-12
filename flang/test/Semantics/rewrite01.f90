! RUN: %flang_fc1 -fdebug-dump-parse-tree %s 2>&1 | FileCheck %s
! Ensure that READ(CVAR) [, item-list] is corrected when CVAR is a
! character variable so as to be a formatted read from the default
! unit, not an unformatted read from an internal unit (which is not
! possible in Fortran).
character :: cvar
! CHECK-NOT: IoUnit -> Variable -> Designator -> DataRef -> Name = 'cvar'
! CHECK: Format -> Expr = 'cvar'
read(cvar)
end

!RUN: %flang_fc1 -fdebug-unparse -pedantic %s  2>&1 | FileCheck %s
!Test rewrite of "PRINT namelistname" into "WRITE(*,NML=namelistname)"
!CHECK: nonstandard: namelist in PRINT statement
namelist /nml/x
x = 123.
!CHECK: WRITE (*, NML=nml)
print nml
end

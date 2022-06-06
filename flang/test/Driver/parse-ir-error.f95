! This file is a valid Fortran file, but we force the driver to treat it as an
! LLVM file (with the `-x` flag). This way we verify that the driver correctly
! rejects invalid LLVM IR input.

!----------
! RUN LINES
!----------
! Input type is implicit (correctly assumed to be Fortran)
! RUN: %flang_fc1 -S %s
! Input type is explicitly set as LLVM IR
! RUN: not %flang -S -x ir %s 2>&1 | FileCheck %s

!----------------
! EXPECTED OUTPUT
!----------------
! CHECK: error: Could not parse IR

end program

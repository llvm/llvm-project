! RUN: not %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck %s

! Test that error compiler directive issues error
program error
!dir$ error "Error!"
!CHECK: error: Error!
end program


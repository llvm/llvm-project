! RUN: %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck %s

! Test that warning compiler directive issues warning
program warn
!dir$ warning "Warning!"
!CHECK: warning: Warning!
end program


! Check that flang driver can handle multiple inputs at once.

! AOCC team xfails this test as its thought to be f18.
! UNSUPPORTED

! RUN: %clang --driver-mode=flang -### -fsyntax-only %S/Inputs/one.f90 %S/Inputs/two.f90 2>&1 | FileCheck --check-prefixes=CHECK-SYNTAX-ONLY %s
! CHECK-SYNTAX-ONLY-LABEL: "{{[^"]*}}flang-new{{[^"/]*}}" "-fc1"
! CHECK-SYNTAX-ONLY: "{{[^"]*}}/Inputs/one.f90"
! CHECK-SYNTAX-ONLY-LABEL: "{{[^"]*}}flang-new{{[^"/]*}}" "-fc1"
! CHECK-SYNTAX-ONLY: "{{[^"]*}}/Inputs/two.f90"

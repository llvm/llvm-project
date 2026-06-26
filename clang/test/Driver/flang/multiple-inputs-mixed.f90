! Check that flang can handle mixed C and fortran inputs.

! RUN: %clang --driver-mode=flang -### -fsyntax-only %S/Inputs/one.f90 %S/Inputs/other.c 2>&1 | FileCheck --check-prefixes=CHECK-SYNTAX-ONLY %s
! CHECK-SYNTAX-ONLY-LABEL: "{{[^"]*}}flang{{[^"/]*}}" "-fc1"
! CHECK-SYNTAX-ONLY: "{{[^"]*}}/Inputs/one.f90"
! CHECK-SYNTAX-ONLY-LABEL: "{{[^"]*}}clang{{[^"/]*}}" "-cc1"
! CHECK-SYNTAX-ONLY: "{{[^"]*}}/Inputs/other.c"

! RUN: not %clang --driver-mode=flang -### -Xflang -std=f2018 %S/Inputs/one.f90 -Xclang -std=c17 %S/Inputs/other.c 2>&1 | FileCheck --check-prefixes=MIXED-OPT %s
! MIXED-OPT: clang: error: unknown argument '-Xclang'
! MIXED-OPT-NOT: "{{[^"]*}}flang{{[^"/]*}}" "-fc1"
! MIXED-OPT-NOT: "-std=f2018"
! MIXED-OPT-NOT: "{{[^"]*}}clang{{[^"/]*}}" "-cc1"
! MIXED-OPT-NOT: "-std=c17"

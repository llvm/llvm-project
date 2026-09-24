! Check that flang can handle mixed C and fortran inputs.

! RUN: %clang --driver-mode=flang -### -fsyntax-only %S/Inputs/one.f90 %S/Inputs/other.c 2>&1 | FileCheck --check-prefixes=CHECK-SYNTAX-ONLY %s
! CHECK-SYNTAX-ONLY-LABEL: "{{[^"]*}}flang{{[^"/]*}}" "-fc1"
! CHECK-SYNTAX-ONLY: "{{[^"]*}}/Inputs/one.f90"
! CHECK-SYNTAX-ONLY-LABEL: "{{[^"]*}}clang{{[^"/]*}}" "-cc1"
! CHECK-SYNTAX-ONLY: "{{[^"]*}}/Inputs/other.c"

! Check that flang-only options are not passed to clang.
! RUN: %clang --driver-mode=flang -### -fstack-arrays %S/Inputs/one.f90 %S/Inputs/other.c 2>&1 | FileCheck --check-prefixes=CHECK-FLANG-OPT %s
! CHECK-FLANG-OPT-LABEL: "{{[^"]*}}flang{{[^"/]*}}" "-fc1"
! CHECK-FLANG-OPT: "-fstack-arrays"
! CHECK-FLANG-OPT-LABEL: "{{[^"]*}}clang{{[^"/]*}}" "-cc1"
! CHECK-FLANG-OPT-NOT: "-fstack-arrays"

! The -std= option is accepted by both clang and flang, but its acceptable values differ between the two.
! Currently, -std=c17 is passed to flang, which rejects it. This should be fixed in the future.
! A potential solution is to use -Xflang and -Xclang to pass the option to the right frontend; however, -Xclang is rejected.

! RUN: %clang --driver-mode=flang -### -std=f2018 %S/Inputs/one.f90 -std=c17 %S/Inputs/other.c 2>&1 | FileCheck --check-prefixes=MIXED-OPT %s
! MIXED-OPT-LABEL: "{{[^"]*}}flang{{[^"/]*}}" "-fc1"
! MIXED-OPT: "-std=f2018"
! MIXED-OPT: "-std=c17"
! MIXED-OPT-LABEL: "{{[^"]*}}clang{{[^"/]*}}" "-cc1"
! MIXED-OPT-NOT: "-std=f2018"
! MIXED-OPT: "-std=c17"

! RUN: not %clang --driver-mode=flang -### -Xflang -std=f2018 %S/Inputs/one.f90 -Xclang -std=c17 %S/Inputs/other.c 2>&1 | FileCheck --check-prefixes=SEPARATE-MIXED-OPT %s
! SEPARATE-MIXED-OPT: error: unknown argument '-Xclang'

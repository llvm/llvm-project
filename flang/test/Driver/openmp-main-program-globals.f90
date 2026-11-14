! The opt-out is FC1-only: -Xflang forwards it, direct driver use is rejected,
! and driver --help omits it.
! REQUIRES: module-independent
! RUN: %flang -### -fopenmp -Xflang -fno-openmp-main-program-globals -S %s 2>&1 | FileCheck %s --check-prefix=FORWARD
! RUN: not %flang -### -fopenmp -fno-openmp-main-program-globals -S %s 2>&1 | FileCheck %s --check-prefix=DRIVER
! RUN: %flang --help | FileCheck %s --check-prefix=HELP --implicit-check-not='-fno-openmp-main-program-globals'
! FORWARD: "-fc1"{{.*}} "-fno-openmp-main-program-globals"
! DRIVER: error: unknown argument '-fno-openmp-main-program-globals';
! DRIVER-SAME: did you mean '-Xflang -fno-openmp-main-program-globals'?
! HELP: OPTIONS:
end

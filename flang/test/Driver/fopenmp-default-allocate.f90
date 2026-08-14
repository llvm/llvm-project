! Check that the driver passes -fopenmp-default-allocate= through to fc1
! and only adds -mmlir -use-alloc-runtime for target mode.

! RUN: %flang -### -S -fopenmp-default-allocate=target %s -o - 2>&1 | FileCheck %s --check-prefix=TARGET
! RUN: %flang -### -S -fopenmp-default-allocate=host %s -o - 2>&1 | FileCheck %s --check-prefix=HOST

! TARGET: warning: -fopenmp-default-allocate= is an experimental feature
! TARGET: "-fc1"
! TARGET-SAME: "-fopenmp-default-allocate=target"
! TARGET-SAME: "-mmlir" "-use-alloc-runtime"

! HOST: warning: -fopenmp-default-allocate= is an experimental feature
! HOST: "-fc1"
! HOST-SAME: "-fopenmp-default-allocate=host"
! HOST-NOT: "-mmlir"
! HOST-NOT: "-use-alloc-runtime"

! Check that invalid values are rejected at the driver level.
! RUN: not %flang -fopenmp-default-allocate=invalid -S %s 2>&1 | FileCheck %s --check-prefix=DRV-INVALID
! DRV-INVALID: error: invalid value 'invalid' in '-fopenmp-default-allocate=invalid'

! Check that invalid values are also rejected at the frontend level.
! RUN: not %flang_fc1 -fopenmp-default-allocate=invalid -S %s 2>&1 | FileCheck %s --check-prefix=FC1-INVALID
! FC1-INVALID: error: invalid value 'invalid' in '-fopenmp-default-allocate=invalid'

program fopenmp_default_allocate
    ! do nothing
end program fopenmp_default_allocate
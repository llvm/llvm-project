! Check that the driver passes -fopenmp-default-allocate= through to fc1
! and only adds -mmlir -use-alloc-runtime for target mode.

! RUN: %flang -### -fopenmp-default-allocate=target %s 2>&1 | FileCheck %s --check-prefix=TARGET
! RUN: %flang -### -fopenmp-default-allocate=host %s 2>&1 | FileCheck %s --check-prefix=HOST

! TARGET: warning: -fopenmp-default-allocate= is an experimental feature
! TARGET: "-fc1"
! TARGET-SAME: "-fopenmp-default-allocate=target"
! TARGET-SAME: "-mmlir" "-use-alloc-runtime"

! HOST: warning: -fopenmp-default-allocate= is an experimental feature
! HOST: "-fc1"
! HOST-SAME: "-fopenmp-default-allocate=host"
! HOST-NOT: "-mmlir"
! HOST-NOT: "-use-alloc-runtime"

! Check that invalid values are rejected at both the driver and frontend level.
! The error message is identical, so a single check prefix is used for both.
! RUN: not %flang -fopenmp-default-allocate=invalid %s 2>&1 | FileCheck %s --check-prefix=INVALID
! RUN: not %flang_fc1 -fopenmp-default-allocate=invalid %s 2>&1 | FileCheck %s --check-prefix=INVALID
! INVALID: error: invalid value 'invalid' in 'fopenmp-default-allocate=', expected one of: target host

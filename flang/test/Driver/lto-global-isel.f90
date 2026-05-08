! -flto passes along an explicit GlobalISel flag.
! RUN: %flang --target=aarch64-linux-gnu -### %s -flto -fglobal-isel 2>&1 \
! RUN:        | FileCheck --check-prefix=CHECK-GISEL %s
! RUN: %flang --target=aarch64-linux-gnu -### %s -flto -fno-global-isel 2>&1 \
! RUN:        | FileCheck --check-prefix=CHECK-DISABLE-GISEL %s
!
! CHECK-GISEL:         "-plugin-opt=-global-isel=1"
! CHECK-DISABLE-GISEL: "-plugin-opt=-global-isel=0"

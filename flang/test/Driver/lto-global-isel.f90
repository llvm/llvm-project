! -flto passes along an explicit GlobalISel flag.
! RUN: %flang --target=aarch64-linux-gnu -### %s -flto -fglobal-isel 2> %t
! RUN: FileCheck --check-prefix=CHECK-GISEL < %t %s
! RUN: %flang --target=aarch64-linux-gnu -### %s -flto -fno-global-isel 2> %t
! RUN: FileCheck --check-prefix=CHECK-DISABLE-GISEL < %t %s
!
! CHECK-GISEL:         "-plugin-opt=-global-isel=1"
! CHECK-DISABLE-GISEL: "-plugin-opt=-global-isel=0"

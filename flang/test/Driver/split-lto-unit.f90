! Check that -fsplit-lto-unit is passed to fc1 by the driver with -fsplit-lto-unit
! RUN: %flang -c -### -fsplit-lto-unit %s 2>&1 | FileCheck %s --check-prefix=SPLIT

! Check that -fsplit-lto-unit is passed to fc1 by the driver with -fsplit-lto-unit
! RUN: %flang -c -### -fno-split-lto-unit -fsplit-lto-unit %s 2>&1 | FileCheck %s  --check-prefix=SPLIT

! Check that -fsplit-lto-unit is not passed to fc1 by the driver with -fno-split-lto-unit
! RUN: %flang -c -### -fno-split-lto-unit %s 2>&1 | FileCheck %s  --check-prefix=NO-SPLIT

! Check that -fsplit-lto-unit is not passed to fc1 by the driver with -fno-split-lto-unit
! RUN: %flang -c -### -fsplit-lto-unit -fno-split-lto-unit %s 2>&1 | FileCheck %s  --check-prefix=NO-SPLIT

! Check that the driver does not pass -fsplit-lto-unit to fc1 by default
! RUN: %flang -c -### %s 2>&1 | FileCheck %s  --check-prefix=NO-SPLIT
! RUN: %flang -c -### -flto %s 2>&1 | FileCheck %s  --check-prefix=NO-SPLIT
! RUN: %flang -c -### -flto=thin %s 2>&1 | FileCheck %s  --check-prefix=NO-SPLIT

! SPLIT: "-fc1"
! SPLIT-SAME: "-fsplit-lto-unit"
! NO-SPLIT-NOT: "-fsplit-lto-unit"

program main
end program main

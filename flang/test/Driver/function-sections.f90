! Test handling of -f(no-)function-sections and -f(no-)data-sections (driver).
!
! CHECK-FS: "-ffunction-sections"
! CHECK-NOFS-NOT: "-ffunction-sections"
! CHECK-DS: "-fdata-sections"
! CHECK-NODS-NOT: "-fdata-sections"

! RUN: %flang -### -target x86_64-none-linux-gnu %s 2>&1 \
! RUN:   | FileCheck %s --check-prefix=CHECK-NOFS --check-prefix=CHECK-NODS

! RUN: %flang -### -target powerpc64-ibm-aix-xcoff %s 2>&1 \
! RUN:   | FileCheck %s --check-prefix=CHECK-NOFS

! RUN: %flang -### %s -ffunction-sections 2>&1 \
! RUN:   | FileCheck %s --check-prefix=CHECK-FS

! RUN: %flang -### %s -fno-function-sections 2>&1 \
! RUN:   | FileCheck %s --check-prefix=CHECK-NOFS

! RUN: %flang -### %s -ffunction-sections -fno-function-sections 2>&1 \
! RUN:   | FileCheck %s --check-prefix=CHECK-NOFS

! RUN: %flang -### %s -fno-function-sections -ffunction-sections 2>&1 \
! RUN:   | FileCheck %s --check-prefix=CHECK-FS

! RUN: %flang -### %s -fdata-sections 2>&1 \
! RUN:   | FileCheck %s --check-prefix=CHECK-DS

! RUN: %flang -### %s -fno-data-sections 2>&1 \
! RUN:   | FileCheck %s --check-prefix=CHECK-NODS

! RUN: %flang -### %s -fdata-sections -fno-data-sections 2>&1 \
! RUN:   | FileCheck %s --check-prefix=CHECK-NODS

! RUN: %flang -### %s -fno-data-sections -fdata-sections 2>&1 \
! RUN:   | FileCheck %s --check-prefix=CHECK-DS

end program

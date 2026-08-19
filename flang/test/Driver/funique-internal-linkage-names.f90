! Test that -funique-internal-linkage-names / -fno-unique-internal-linkage-names are forwarded to flang -fc1.

! RUN: %flang -### %s 2>&1 | FileCheck %s --check-prefix=DISABLED
! RUN: %flang -### -funique-internal-linkage-names %s 2>&1 | FileCheck %s --check-prefix=ENABLED
! RUN: %flang -### -fno-unique-internal-linkage-names %s 2>&1 | FileCheck %s --check-prefix=DISABLED
! RUN: %flang -### -funique-internal-linkage-names -fno-unique-internal-linkage-names %s 2>&1 | FileCheck %s --check-prefix=DISABLED
! RUN: %flang -### -fno-unique-internal-linkage-names -funique-internal-linkage-names %s 2>&1 | FileCheck %s --check-prefix=ENABLED

! DISABLED: "-fc1"
! DISABLED-NOT: "-funique-internal-linkage-names"
! DISABLED-NOT: "-fno-unique-internal-linkage-names"

! ENABLED: "-fc1"
! ENABLED-SAME: "-funique-internal-linkage-names"


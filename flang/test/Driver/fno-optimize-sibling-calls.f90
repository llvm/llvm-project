! Test -f[no-]optimize-sibling-calls driver forwarding to flang -fc1.

! RUN: %flang -### %s 2>&1 | FileCheck %s --check-prefix=OPTIMIZE-SIBLING
! RUN: %flang -### -foptimize-sibling-calls %s 2>&1 | FileCheck %s --check-prefix=OPTIMIZE-SIBLING
! RUN: %flang -### -fno-optimize-sibling-calls %s 2>&1 | FileCheck %s --check-prefix=NO-OPTIMIZE-SIBLING
! RUN: %flang -### -fno-optimize-sibling-calls -foptimize-sibling-calls %s 2>&1 | FileCheck %s --check-prefix=OPTIMIZE-SIBLING
! RUN: %flang -### -foptimize-sibling-calls -fno-optimize-sibling-calls %s 2>&1 | FileCheck %s --check-prefix=NO-OPTIMIZE-SIBLING

! OPTIMIZE-SIBLING: "-fc1"
! OPTIMIZE-SIBLING-NOT: "-fno-optimize-sibling-calls"

! NO-OPTIMIZE-SIBLING: "-fc1"
! NO-OPTIMIZE-SIBLING-SAME: "-fno-optimize-sibling-calls"

! Test -f[no-]optimize-sibling-calls driver forwarding to flang -fc1.

! RUN: %flang -### %s 2>&1 | FileCheck %s --check-prefixes=FC1,OPTIMIZE-SIBLING
! RUN: %flang -### -foptimize-sibling-calls %s 2>&1 | FileCheck %s --check-prefixes=FC1,OPTIMIZE-SIBLING
! RUN: %flang -### -fno-optimize-sibling-calls %s 2>&1 | FileCheck %s --check-prefixes=FC1,NO-OPTIMIZE-SIBLING
! RUN: %flang -### -fno-optimize-sibling-calls -foptimize-sibling-calls %s 2>&1 | FileCheck %s --check-prefixes=FC1,OPTIMIZE-SIBLING
! RUN: %flang -### -foptimize-sibling-calls -fno-optimize-sibling-calls %s 2>&1 | FileCheck %s --check-prefixes=FC1,NO-OPTIMIZE-SIBLING

! FC1: "-fc1"
! OPTIMIZE-SIBLING-NOT: "-fno-optimize-sibling-calls"
! NO-OPTIMIZE-SIBLING-SAME: "-fno-optimize-sibling-calls"

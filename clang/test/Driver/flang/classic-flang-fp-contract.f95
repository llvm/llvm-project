! XFAIL: *
! REQUIRES: classic_flang

! RUN: %flang -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-FLANG2-FP-CONTRACT-ABSENCE
! RUN: %flang -O1 -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-FLANG2-FP-CONTRACT
! RUN: %flang -O2 -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-FLANG2-FP-CONTRACT
! RUN: %flang -O3 -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-FLANG2-FP-CONTRACT
! RUN: %flang -Ofast -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-FLANG2-FP-CONTRACT
! RUN: %flang -ffp-contract=fast -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-FLANG2-FP-CONTRACT-ABSENCE
! RUN: %flang -O1 -ffp-contract=fast -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-FLANG2-FP-CONTRACT
! RUN: %flang -O2 -ffp-contract=fast -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-FLANG2-FP-CONTRACT
! RUN: %flang -O3 -ffp-contract=fast -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-FLANG2-FP-CONTRACT
! RUN: %flang -Ofast -ffp-contract=fast -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-FLANG2-FP-CONTRACT
! RUN: %flang -ffp-contract=on -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-FLANG2-FP-CONTRACT-ABSENCE
! RUN: %flang -O1 -ffp-contract=on -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-FLANG2-FP-CONTRACT
! RUN: %flang -O2 -ffp-contract=on -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-FLANG2-FP-CONTRACT
! RUN: %flang -O3 -ffp-contract=on -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-FLANG2-FP-CONTRACT
! RUN: %flang -Ofast -ffp-contract=on -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-FLANG2-FP-CONTRACT
! RUN: %flang -ffp-contract=off -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-FLANG2-FP-CONTRACT-ABSENCE
! RUN: %flang -O1 -ffp-contract=off -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-FLANG2-FP-CONTRACT-ABSENCE
! RUN: %flang -O2 -ffp-contract=off -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-FLANG2-FP-CONTRACT-ABSENCE
! RUN: %flang -O3 -ffp-contract=off -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-FLANG2-FP-CONTRACT-ABSENCE
! RUN: %flang -Ofast -ffp-contract=off -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-FLANG2-FP-CONTRACT-ABSENCE

! CHECK-FLANG2-FP-CONTRACT: "{{.*}}flang2"
! CHECK-FLANG2-FP-CONTRACT-SAME: "-x" "172" "0x40000000" "-x" "179" "1" "-x" "216" "0x1000"
! CHECK-FLANG2-FP-CONTRACT-ABSENCE: "{{.*}}flang2"
! CHECK-FLANG2-FP-CONTRACT-ABSENCE-SAME: "-x" "171" "0x40000000" "-x" "178" "1"

! REQUIRES: classic_flang

! Check that the driver invokes flang1 correctly for preprocessed fixed-form
! Fortran code.

! RUN: %clang --driver-mode=flang -target x86_64-unknown-linux-gnu -c %s -### 2>&1 \
! RUN:   | FileCheck %s
! CHECK: "{{.*}}flang1"
! CHECK-NOT: "-preprocess"
! CHECK-SAME: "-nofreeform"
! CHECK-NEXT: "{{.*}}flang2"
! CHECK-NEXT: {{clang.* "-cc1"}}

! Check that the driver invokes flang1 correctly when preprocessing is
! explicitly requested.

! RUN: %clang --driver-mode=flang -target x86_64-unknown-linux-gnu -E %s -### 2>&1 \
! RUN:   | FileCheck --check-prefix=CHECK-PREPROCESS %s
! CHECK-PREPROCESS: "{{.*}}flang1"
! CHECK-PREPROCESS-SAME: "-preprocess"
! CHECK-PREPROCESS-SAME: "-es"
! CHECK-PREPROCESS-SAME: "-pp"
! CHECK-PREPROCESS-NOT: "{{.*}}flang1"
! CHECK-PREPROCESS-NOT: "{{.*}}flang2"
! CHECK-PREPROCESS-NOT: {{clang.* "-cc1"}}
! CHECK-PREPROCESS-NOT: {{clang.* "-cc1as"}}

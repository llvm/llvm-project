! REQUIRES: classic_flang

! Check that the driver invokes flang1 correctly for free-form Fortran code
! which requires preprocessing.

! RUN: %clang --driver-mode=flang -target x86_64-unknown-linux-gnu -c %s -### 2>&1 \
! RUN:   | FileCheck %s
! CHECK: "{{.*}}flang1"
! CHECK-SAME: "-preprocess"
! CHECK-SAME: "-freeform"
! CHECK-NEXT: "{{.*}}flang2"
! CHECK-NEXT: {{clang.* "-cc1"}}

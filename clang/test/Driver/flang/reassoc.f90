! REQUIRES: classic_flang

! UNSUPPORTED: classic_flang

! Tests for flags which generate nsw, reassoc attributes

! RUN: %flang -Kieee %s -### 2>&1 | FileCheck --check-prefixes=IEEE,NO_FAST,NO_RELAXED,NO_REASSOC,NO_NSZ %s
! RUN: %flang -Knoieee %s -### 2>&1 | FileCheck --check-prefixes=NO_IEEE,NO_FAST,NO_RELAXED,NO_REASSOC,NO_NSZ %s
! RUN: %flang -ffast-math %s -### 2>&1 | FileCheck --check-prefixes=NO_IEEE,FAST,NO_RELAXED,NO_REASSOC,NO_NSZ %s
! RUN: %flang -fno-fast-math %s -### 2>&1 | FileCheck --check-prefixes=NO_FAST,NO_RELAXED,NO_REASSOC,NO_NSZ %s
! RUN: %flang -frelaxed-math %s -### 2>&1 | FileCheck --check-prefixes=NO_IEEE,NO_FAST,RELAXED,NO_REASSOC,NO_NSZ %s
! RUN: %flang -fassociative-math %s -### 2>&1 | FileCheck --check-prefixes=NO_IEEE,NO_FAST,NO_RELAXED,REASSOC_NSZ %s
! RUN: %flang -fno-associative-math %s -### 2>&1 | FileCheck --check-prefixes=NO_FAST,NO_RELAXED,NO_REASSOC,NO_NSZ %s
! RUN: %flang -fsigned-zeros %s -### 2>&1 | FileCheck --check-prefixes=NO_FAST,NO_RELAXED,NO_REASSOC,NO_NSZ %s
! RUN: %flang -fno-signed-zeros %s -### 2>&1 | FileCheck --check-prefixes=NO_IEEE,NO_FAST,NO_RELAXED,NO_REASSOC,NSZ %s

! RUN: %flang -fno-associative-math -fno-signed-zeros %s -### 2>&1 | FileCheck --check-prefixes=NO_IEEE,NO_FAST,NO_RELAXED,NO_REASSOC,NSZ %s
! RUN: %flang -fno-associative-math -fsigned-zeros %s -### 2>&1 | FileCheck --check-prefixes=NO_FAST,NO_RELAXED,NO_REASSOC,NO_NSZ %s
! RUN: %flang -fassociative-math -fno-signed-zeros %s -### 2>&1 | FileCheck --check-prefixes=NO_IEEE,NO_FAST,NO_RELAXED,REASSOC_NSZ %s
! RUN: %flang -fassociative-math -fsigned-zeros %s -### 2>&1 | FileCheck --check-prefixes=NO_IEEE,NO_FAST,NO_RELAXED,NO_REASSOC,NO_NSZ %s

! RUN: %flang -Kieee -fassociative-math %s -### 2>&1 | FileCheck --check-prefixes=NO_IEEE,NO_FAST,NO_RELAXED,REASSOC_NSZ %s
! RUN: %flang -Kieee -fno-associative-math %s -### 2>&1 | FileCheck --check-prefixes=IEEE,NO_FAST,NO_RELAXED,NO_REASSOC,NO_NSZ %s
! RUN: %flang -Kieee -fsigned-zeros %s -### 2>&1 | FileCheck --check-prefixes=IEEE,NO_FAST,NO_RELAXED,NO_REASSOC,NO_NSZ %s
! RUN: %flang -Kieee -fno-signed-zeros %s -### 2>&1 | FileCheck --check-prefixes=NO_IEEE,NO_FAST,NO_RELAXED,NO_REASSOC,NSZ %s
! RUN: %flang -ffast-math -fassociative-math %s -### 2>&1 | FileCheck --check-prefixes=NO_IEEE,NO_RELAXED,REASSOC_NSZ %s
! RUN: %flang -ffast-math -fno-associative-math %s -### 2>&1 | FileCheck --check-prefixes=NO_IEEE,FAST,NO_RELAXED,NO_REASSOC,NO_NSZ %s
! RUN: %flang -ffast-math -fsigned-zeros %s -### 2>&1 | FileCheck --check-prefixes=NO_IEEE,FAST,NO_RELAXED,NO_REASSOC,NO_NSZ %s
! RUN: %flang -ffast-math -fno-signed-zeros %s -### 2>&1 | FileCheck --check-prefixes=NO_IEEE,NO_RELAXED,NO_REASSOC,NSZ %s
! RUN: %flang -frelaxed-math -fassociative-math %s -### 2>&1 | FileCheck --check-prefixes=NO_IEEE,NO_FAST,REASSOC_NSZ %s
! RUN: %flang -frelaxed-math -fno-associative-math %s -### 2>&1 | FileCheck --check-prefixes=NO_IEEE,NO_FAST,RELAXED,NO_REASSOC,NO_NSZ %s
! RUN: %flang -frelaxed-math -fsigned-zeros %s -### 2>&1 | FileCheck --check-prefixes=NO_IEEE,NO_FAST,RELAXED,NO_REASSOC,NO_NSZ %s
! RUN: %flang -frelaxed-math -fno-signed-zeros %s -### 2>&1 | FileCheck --check-prefixes=NO_IEEE,NO_FAST,NO_REASSOC,NSZ %s

! RUN: %flang -fassociative-math -Kieee %s -### 2>&1 | FileCheck --check-prefixes=IEEE,NO_FAST,NO_RELAXED,NO_REASSOC,NO_NSZ %s
! RUN: %flang -fno-associative-math -Kieee %s -### 2>&1 | FileCheck --check-prefixes=IEEE,NO_FAST,NO_RELAXED,NO_REASSOC,NO_NSZ %s
! RUN: %flang -fsigned-zeros -Kieee %s -### 2>&1 | FileCheck --check-prefixes=IEEE,NO_FAST,NO_RELAXED,NO_REASSOC,NO_NSZ %s
! RUN: %flang -fno-signed-zeros -Kieee %s -### 2>&1 | FileCheck --check-prefixes=IEEE,NO_FAST,NO_RELAXED,NO_REASSOC,NO_NSZ %s
! RUN: %flang -fassociative-math -ffast-math %s -### 2>&1 | FileCheck --check-prefixes=NO_IEEE,NO_RELAXED,REASSOC_NSZ %s
! RUN: %flang -fno-associative-math -ffast-math %s -### 2>&1 | FileCheck --check-prefixes=NO_IEEE,FAST,NO_RELAXED,NO_REASSOC,NO_NSZ %s
! RUN: %flang -fsigned-zeros -ffast-math %s -### 2>&1 | FileCheck --check-prefixes=NO_IEEE,FAST,NO_RELAXED,NO_REASSOC,NO_NSZ %s
! RUN: %flang -fno-signed-zeros -ffast-math %s -### 2>&1 | FileCheck --check-prefixes=NO_IEEE,NO_RELAXED,NO_REASSOC,NSZ %s
! RUN: %flang -fassociative-math -frelaxed-math %s -### 2>&1 | FileCheck --check-prefixes=NO_IEEE,NO_FAST,REASSOC_NSZ %s
! RUN: %flang -fno-associative-math -frelaxed-math %s -### 2>&1 | FileCheck --check-prefixes=NO_IEEE,NO_FAST,RELAXED,NO_REASSOC,NO_NSZ %s
! RUN: %flang -fsigned-zeros -frelaxed-math %s -### 2>&1 | FileCheck --check-prefixes=NO_IEEE,NO_FAST,RELAXED,NO_REASSOC,NO_NSZ %s
! RUN: %flang -fno-signed-zeros -frelaxed-math %s -### 2>&1 | FileCheck --check-prefixes=NO_IEEE,NO_FAST,NO_REASSOC,NSZ %s

! IEEE: {{.*}}flang2{{.*}} "-ieee" "1"
! NO_IEEE-NOT: {{.*}}flang2{{.*}} "-ieee" "1"

! FAST: {{.*}}flang2{{.*}} "-x" "216" "1"
! NO_FAST-NOT: {{.*}}flang2{{.*}} "-x" "216" "1"

! RELAXED: {{.*}}flang2{{.*}} "-x" "15" "0x400"
! NO_RELAXED-NOT: {{.*}}flang2{{.*}} "-x" "15" "0x400"

! REASSOC_NSZ: {{.*}}flang2{{.*}} "-x" "216" "0x8" "-x" "216" "0x10"
! NO_REASSOC-NOT: {{.*}}flang2{{.*}} "-x" "216" "0x10"

! NSZ: {{.*}}flang2{{.*}} "-x" "216" "0x8"
! NO_NSZ-NOT: {{.*}}flang2{{.*}} "-x" "216" "0x8"

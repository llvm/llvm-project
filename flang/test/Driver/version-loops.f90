! Test that flang forwards the -f{no-,}version-loops-for-stride 
! options correctly to flang -fc1 for different variants of optimisation
! and explicit flags.

! RUN: %flang -### %s -o %t 2>&1   -O3 \
! RUN:   | FileCheck %s
  
! RUN: %flang -### %s -o %t 2>&1 -O2 \
! RUN:   | FileCheck %s --check-prefix=CHECK-O2

! RUN: %flang -### %s -o %t 2>&1  -O2 -fversion-loops-for-stride \
! RUN:   | FileCheck %s --check-prefix=CHECK-O2-with
  
! RUN: %flang -### %s -o %t 2>&1  -O4 \
! RUN:   | FileCheck %s --check-prefix=CHECK-O4
  
! RUN: %flang -### %s -o %t 2>&1  -Ofast \
! RUN:   | FileCheck %s --check-prefix=CHECK-Ofast
  
! RUN: %flang -### %s -o %t 2>&1 -Ofast -fno-version-loops-for-stride \
! RUN:   | FileCheck %s --check-prefix=CHECK-Ofast-no

! RUN: %flang -### %s -o %t 2>&1 -O3 -fno-version-loops-for-stride \
! RUN:   | FileCheck %s --check-prefix=CHECK-O3-no

! CHECK: "{{.*}}flang" "-fc1"
! CHECK-SAME: "-fversion-loops-for-stride"
! CHECK-SAME: "-O3"

! CHECK-O2: "{{.*}}flang" "-fc1"
! CHECK-O2-NOT: "-fversion-loops-for-stride"
! CHECK-O2-SAME: "-O2"  

! CHECK-O2-with: "{{.*}}flang" "-fc1"
! CHECK-O2-with-SAME: "-fversion-loops-for-stride"
! CHECK-O2-with-SAME: "-O2"  
  
! CHECK-O4: "{{.*}}flang" "-fc1"
! CHECK-O4-SAME: "-fversion-loops-for-stride"
! CHECK-O4-SAME: "-O3"

! CHECK-Ofast: "{{.*}}flang" "-fc1"
! CHECK-Ofast-SAME: "-ffast-math"
! CHECK-Ofast-SAME: "-fversion-loops-for-stride"
! CHECK-Ofast-SAME: "-O3"

! CHECK-Ofast-no: "{{.*}}flang" "-fc1"
! CHECK-Ofast-no-SAME: "-ffast-math"
! CHECK-Ofast-no-NOT: "-fversion-loops-for-stride"
! CHECK-Ofast-no-SAME: "-O3"

! CHECK-O3-no: "{{.*}}flang" "-fc1"
! CHECK-O3-no-NOT: "-fversion-loops-for-stride"
! CHECK-O3-no-SAME: "-O3"

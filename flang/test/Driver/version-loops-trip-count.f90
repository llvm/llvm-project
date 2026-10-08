! Test that flang forwards -f{no-,}version-loops-for-trip-count to flang -fc1.
! The flag follows the same -O level rule as -fversion-loops-for-stride (on at
! -O3 and above, explicit flag wins), which version-loops.f90 covers in full;
! this checks the new flag and that the two flags are independent.

! RUN: %flang -### %s -o %t 2>&1 -O3 \
! RUN:   | FileCheck %s --check-prefix=CHECK-O3

! RUN: %flang -### %s -o %t 2>&1 -O3 -fno-version-loops-for-trip-count \
! RUN:   | FileCheck %s --check-prefix=CHECK-O3-no

! RUN: %flang -### %s -o %t 2>&1 -O2 -fversion-loops-for-trip-count \
! RUN:   | FileCheck %s --check-prefix=CHECK-O2-with

! RUN: %flang -### %s -o %t 2>&1 -O3 -fno-version-loops-for-stride \
! RUN:   | FileCheck %s --check-prefix=CHECK-O3-no-stride

! CHECK-O3: "{{.*}}flang{{.*}}" "-fc1"
! CHECK-O3-SAME: "-fversion-loops-for-trip-count"
! CHECK-O3-SAME: "-O3"

! CHECK-O3-no: "{{.*}}flang{{.*}}" "-fc1"
! CHECK-O3-no-NOT: "-fversion-loops-for-trip-count"
! CHECK-O3-no-SAME: "-O3"

! CHECK-O2-with: "{{.*}}flang{{.*}}" "-fc1"
! CHECK-O2-with-SAME: "-fversion-loops-for-trip-count"
! CHECK-O2-with-SAME: "-O2"

! Disabling stride versioning leaves trip-count versioning on.
! CHECK-O3-no-stride: "{{.*}}flang{{.*}}" "-fc1"
! CHECK-O3-no-stride-NOT: "-fversion-loops-for-stride"
! CHECK-O3-no-stride-SAME: "-fversion-loops-for-trip-count"
! CHECK-O3-no-stride-SAME: "-O3"

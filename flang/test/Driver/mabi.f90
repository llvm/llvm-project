! RUN: not %flang -### -c --target=powerpc64le-unknown-linux -mabi=vec-extabi %s 2>&1 | FileCheck --check-prefix=INVALID1 %s
! RUN: not %flang -### -c --target=x86_64-unknown-linux -mabi=vec-extabi %s 2>&1 | FileCheck --check-prefix=INVALID2 %s
! RUN: %flang -### -c -target powerpc-unknown-aix %s 2>&1 | FileCheck --implicit-check-not=vec-extabi %s
! RUN: %flang -### -c -target powerpc-unknown-aix -mabi=vec-default %s 2>&1 | FileCheck --implicit-check-not=vec-extabi %s
! RUN: %flang -### -c -target powerpc-unknown-aix -mabi=vec-extabi %s 2>&1 | FileCheck --check-prefix=EXTABI %s

! INVALID1: error: unsupported option '-mabi=vec-extabi' for target '{{.*}}'
! INVALID2: error: unsupported option '-mabi=' for target '{{.*}}'

! EXTABI: "-fc1"
! EXTABI-SAME: "-mabi=vec-extabi"



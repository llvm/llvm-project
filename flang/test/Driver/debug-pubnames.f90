! Test that -gpubnames and -ggnu-pubnames are passed on to the frontend, and
! that the last of these and their -gno- forms wins.

! RUN: %flang -### -c -target x86_64 -g %s 2>&1 \
! RUN:   | FileCheck %s --check-prefix=NONE
! RUN: %flang -### -c -target x86_64 -g -gpubnames %s 2>&1 \
! RUN:   | FileCheck %s --check-prefix=PUB
! RUN: %flang -### -c -target x86_64 -g -ggnu-pubnames %s 2>&1 \
! RUN:   | FileCheck %s --check-prefix=GNU
! RUN: %flang -### -c -target x86_64 -g -ggnu-pubnames -gpubnames %s 2>&1 \
! RUN:   | FileCheck %s --check-prefix=PUB
! RUN: %flang -### -c -target x86_64 -g -gpubnames -ggnu-pubnames %s 2>&1 \
! RUN:   | FileCheck %s --check-prefix=GNU

! The -gno- forms are accepted on their own, and each cancels both requests.
! RUN: %flang -### -c -target x86_64 -g -gno-pubnames %s 2>&1 \
! RUN:   | FileCheck %s --check-prefix=NONE
! RUN: %flang -### -c -target x86_64 -g -gno-gnu-pubnames %s 2>&1 \
! RUN:   | FileCheck %s --check-prefix=NONE
! RUN: %flang -### -c -target x86_64 -g -gpubnames -gno-pubnames %s 2>&1 \
! RUN:   | FileCheck %s --check-prefix=NONE
! RUN: %flang -### -c -target x86_64 -g -gpubnames -gno-gnu-pubnames %s \
! RUN:   2>&1 | FileCheck %s --check-prefix=NONE
! RUN: %flang -### -c -target x86_64 -g -ggnu-pubnames -gno-gnu-pubnames %s \
! RUN:   2>&1 | FileCheck %s --check-prefix=NONE
! RUN: %flang -### -c -target x86_64 -g -ggnu-pubnames -gno-pubnames %s 2>&1 \
! RUN:   | FileCheck %s --check-prefix=NONE
! RUN: %flang -### -c -target x86_64 -g -gno-pubnames -gpubnames %s 2>&1 \
! RUN:   | FileCheck %s --check-prefix=PUB
! RUN: %flang -### -c -target x86_64 -g -gno-gnu-pubnames -ggnu-pubnames %s \
! RUN:   2>&1 | FileCheck %s --check-prefix=GNU

! NONE-NOT: error:
! NONE: "-fc1"
! NONE-NOT: "-gpubnames"
! NONE-NOT: "-ggnu-pubnames"

! PUB-NOT: error:
! PUB: "-fc1"
! PUB-NOT: "-ggnu-pubnames"
! PUB-SAME: "-gpubnames"
! PUB-NOT: "-ggnu-pubnames"

! GNU-NOT: error:
! GNU: "-fc1"
! GNU-NOT: "-gpubnames"
! GNU-SAME: "-ggnu-pubnames"
! GNU-NOT: "-gpubnames"

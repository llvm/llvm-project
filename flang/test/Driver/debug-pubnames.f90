! Test that -gpubnames and -ggnu-pubnames are passed on to the frontend, and
! that the last of these and their -gno- forms wins.

! RUN: %flang -### -target x86_64 -g %s 2>&1 \
! RUN:   | FileCheck %s --check-prefix=NONE
! RUN: %flang -### -target x86_64 -g -gpubnames %s 2>&1 \
! RUN:   | FileCheck %s --check-prefix=PUB
! RUN: %flang -### -target x86_64 -g -ggnu-pubnames %s 2>&1 \
! RUN:   | FileCheck %s --check-prefix=GNU
! RUN: %flang -### -target x86_64 -g -ggnu-pubnames -gpubnames %s 2>&1 \
! RUN:   | FileCheck %s --check-prefix=PUB
! RUN: %flang -### -target x86_64 -g -gpubnames -ggnu-pubnames %s 2>&1 \
! RUN:   | FileCheck %s --check-prefix=GNU

! The -gno- forms are accepted on their own, and each cancels both requests.
! RUN: %flang -### -target x86_64 -g -gno-pubnames %s 2>&1 \
! RUN:   | FileCheck %s --check-prefix=NONE
! RUN: %flang -### -target x86_64 -g -gno-gnu-pubnames %s 2>&1 \
! RUN:   | FileCheck %s --check-prefix=NONE
! RUN: %flang -### -target x86_64 -g -gpubnames -gno-pubnames %s 2>&1 \
! RUN:   | FileCheck %s --check-prefix=NONE
! RUN: %flang -### -target x86_64 -g -gpubnames -gno-gnu-pubnames %s 2>&1 \
! RUN:   | FileCheck %s --check-prefix=NONE
! RUN: %flang -### -target x86_64 -g -ggnu-pubnames -gno-gnu-pubnames %s 2>&1 \
! RUN:   | FileCheck %s --check-prefix=NONE
! RUN: %flang -### -target x86_64 -g -ggnu-pubnames -gno-pubnames %s 2>&1 \
! RUN:   | FileCheck %s --check-prefix=NONE
! RUN: %flang -### -target x86_64 -g -gno-pubnames -gpubnames %s 2>&1 \
! RUN:   | FileCheck %s --check-prefix=PUB
! RUN: %flang -### -target x86_64 -g -gno-gnu-pubnames -ggnu-pubnames %s 2>&1 \
! RUN:   | FileCheck %s --check-prefix=GNU

! Split DWARF implies -ggnu-pubnames, unless one of the flags says otherwise.
! RUN: %flang -### -target x86_64 -g -gsplit-dwarf %s 2>&1 \
! RUN:   | FileCheck %s --check-prefix=GNU
! RUN: %flang -### -target x86_64 -g -gsplit-dwarf=single %s 2>&1 \
! RUN:   | FileCheck %s --check-prefix=GNU
! RUN: %flang -### -target x86_64 -g -gsplit-dwarf -gpubnames %s 2>&1 \
! RUN:   | FileCheck %s --check-prefix=PUB
! RUN: %flang -### -target x86_64 -g -gsplit-dwarf -gno-pubnames %s 2>&1 \
! RUN:   | FileCheck %s --check-prefix=NONE
! RUN: %flang -### -target x86_64 -g -gsplit-dwarf -gno-gnu-pubnames %s 2>&1 \
! RUN:   | FileCheck %s --check-prefix=NONE

! It does not when split DWARF ends up off: without -g, with -gno-split-dwarf,
! or on Darwin.
! RUN: %flang -### -target x86_64 -gsplit-dwarf %s 2>&1 \
! RUN:   | FileCheck %s --check-prefix=NONE
! RUN: %flang -### -target x86_64 -g -gsplit-dwarf -gno-split-dwarf %s 2>&1 \
! RUN:   | FileCheck %s --check-prefix=NONE
! RUN: %flang -### -target x86_64-apple-darwin -g -gsplit-dwarf %s 2>&1 \
! RUN:   | FileCheck %s --check-prefix=NONE

! NONE: "-fc1"
! NONE-NOT: "-gpubnames"
! NONE-NOT: "-ggnu-pubnames"

! PUB: "-fc1"
! PUB-SAME: "-gpubnames"

! GNU: "-fc1"
! GNU-SAME: "-ggnu-pubnames"

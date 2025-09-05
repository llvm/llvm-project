! Test range options for complex multiplication and division.

! RUN: %flang -### -c %s 2>&1 \
! RUN:   | FileCheck %s --check-prefix=RANGE

! RUN: %flang -### -fcomplex-arithmetic=full -c %s 2>&1 \
! RUN:   | FileCheck %s --check-prefix=FULL

! RUN: %flang -### -fcomplex-arithmetic=improved -c %s 2>&1 \
! RUN:   | FileCheck %s --check-prefix=IMPRVD

! RUN: %flang -### -fcomplex-arithmetic=basic -c %s 2>&1 \
! RUN:   | FileCheck %s --check-prefix=BASIC

! RUN: not %flang -### -fcomplex-arithmetic=foo -c %s 2>&1 \
! RUN:   | FileCheck %s --check-prefix=ERR

! RANGE-NOT: -complex-range=
! FULL: -complex-range=full
! IMPRVD: -complex-range=improved
! BASIC: -complex-range=basic

! ERR: error: unsupported argument 'foo' to option '-fcomplex-arithmetic='

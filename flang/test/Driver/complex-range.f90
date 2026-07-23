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

! RUN: %flang -### -ffast-math -c %s 2>&1 \
! RUN:   | FileCheck %s --check-prefix=BASIC

! RUN: %flang -### -fno-fast-math -c %s 2>&1 \
! RUN:   | FileCheck %s --check-prefix=RANGE

! RUN: %flang -### -Werror -ffast-math -fno-fast-math -c %s 2>&1 \
! RUN:   | FileCheck --check-prefixes=RANGE %s

! RUN: %flang -### -ffast-math -fcomplex-arithmetic=full -c %s 2>&1 \
! RUN:   | FileCheck --check-prefixes=FULL,ARITH-FULL-OVERRIDING,FAST-OVERRIDDEN %s

! RUN: %flang -### -ffast-math -fcomplex-arithmetic=improved -c %s 2>&1 \
! RUN:   | FileCheck --check-prefixes=IMPRVD,ARITH-IMPROVED-OVERRIDING,FAST-OVERRIDDEN %s

! RUN: %flang -### -Werror -ffast-math -fcomplex-arithmetic=basic -c %s 2>&1 \
! RUN:   | FileCheck --check-prefixes=BASIC %s

! RUN: %flang -### -Werror -fno-fast-math -ffast-math -c %s 2>&1 \
! RUN:   | FileCheck --check-prefixes=BASIC %s

! RUN: %flang -### -Werror -fno-fast-math -fcomplex-arithmetic=full -c %s 2>&1 \
! RUN:   | FileCheck --check-prefixes=FULL %s

! RUN: %flang -### -Werror -fno-fast-math -fcomplex-arithmetic=improved -c %s 2>&1 \
! RUN:   | FileCheck --check-prefixes=IMPRVD %s

! RUN: %flang -### -Werror -fno-fast-math -fcomplex-arithmetic=basic -c %s 2>&1 \
! RUN:   | FileCheck --check-prefixes=BASIC %s

! RUN: %flang -### -fcomplex-arithmetic=full -ffast-math -c %s 2>&1 \
! RUN:   | FileCheck --check-prefixes=BASIC,FAST-OVERRIDING,ARITH-FULL-OVERRIDDEN %s

! RUN: %flang -### -Werror -fcomplex-arithmetic=full -fno-fast-math -c %s 2>&1 \
! RUN:   | FileCheck --check-prefixes=RANGE %s

! RUN: %flang -### -Werror -fcomplex-arithmetic=full -fcomplex-arithmetic=improved -c %s 2>&1 \
! RUN:   | FileCheck --check-prefixes=IMPRVD %s

! RUN: %flang -### -Werror -fcomplex-arithmetic=full -fcomplex-arithmetic=basic -c %s 2>&1 \
! RUN:   | FileCheck --check-prefixes=BASIC %s

! RUN: %flang -### -fcomplex-arithmetic=improved -ffast-math -c %s 2>&1 \
! RUN:   | FileCheck --check-prefixes=BASIC,FAST-OVERRIDING,ARITH-IMPROVED-OVERRIDDEN %s

! RUN: %flang -### -fcomplex-arithmetic=improved -fno-fast-math -c %s 2>&1 \
! RUN:   | FileCheck --check-prefixes=RANGE,NOFAST-OVERRIDING,ARITH-IMPROVED-OVERRIDDEN %s

! RUN: %flang -### -Werror -fcomplex-arithmetic=improved -fcomplex-arithmetic=full -c %s 2>&1 \
! RUN:   | FileCheck --check-prefixes=FULL %s

! RUN: %flang -### -Werror -fcomplex-arithmetic=improved -fcomplex-arithmetic=basic -c %s 2>&1 \
! RUN:   | FileCheck --check-prefixes=BASIC %s

! RUN: %flang -### -Werror -fcomplex-arithmetic=basic -ffast-math -c %s 2>&1 \
! RUN:   | FileCheck --check-prefixes=BASIC %s

! RUN: %flang -### -fcomplex-arithmetic=basic -fno-fast-math -c %s 2>&1 \
! RUN:   | FileCheck --check-prefixes=RANGE,NOFAST-OVERRIDING,ARITH-BASIC-OVERRIDDEN %s

! RUN: %flang -### -Werror -fcomplex-arithmetic=basic -fcomplex-arithmetic=full -c %s 2>&1 \
! RUN:   | FileCheck --check-prefixes=FULL %s

! RUN: %flang -### -Werror -fcomplex-arithmetic=basic -fcomplex-arithmetic=improved -c %s 2>&1 \
! RUN:   | FileCheck --check-prefixes=IMPRVD %s


! FAST-OVERRIDING: warning: '-ffast-math' sets complex range to "basic"
! NOFAST-OVERRIDING: warning: '-fno-fast-math' sets complex range to "none"
! ARITH-FULL-OVERRIDING: warning: '-fcomplex-arithmetic=full' sets complex range to "full"
! ARITH-IMPROVED-OVERRIDING: warning: '-fcomplex-arithmetic=improved' sets complex range to "improved"

! FAST-OVERRIDDEN: overriding the setting of "basic" that was implied by '-ffast-math' [-Woverriding-complex-range]
! ARITH-FULL-OVERRIDDEN: overriding the setting of "full" that was implied by '-fcomplex-arithmetic=full' [-Woverriding-complex-range]
! ARITH-IMPROVED-OVERRIDDEN: overriding the setting of "improved" that was implied by '-fcomplex-arithmetic=improved' [-Woverriding-complex-range]
! ARITH-BASIC-OVERRIDDEN: overriding the setting of "basic" that was implied by '-fcomplex-arithmetic=basic' [-Woverriding-complex-range]

! RANGE-NOT: -complex-range=
! FULL: -complex-range=full
! IMPRVD: -complex-range=improved
! BASIC: -complex-range=basic

! ERR: error: unsupported argument 'foo' to option '-fcomplex-arithmetic='

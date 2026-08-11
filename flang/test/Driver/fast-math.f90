! Test for correct forwarding of fast-math flags from the compiler driver to the
! frontend driver

! Check warning message for Ofast deprecation
! RUN: %flang -Ofast -### %s -o %t 2>&1 | FileCheck %s
! CHECK: warning: argument '-Ofast' is deprecated; use '-O3 -ffast-math -fstack-arrays -fno-protect-parens' for the same behavior, or '-O3 -fstack-arrays' to enable only conforming optimizations [-Wdeprecated-ofast]

! -Ofast => -ffast-math -O3 -fstack-arrays
! RUN: %flang -Ofast -fsyntax-only -### %s -o %t 2>&1 \
! RUN:     | FileCheck --check-prefix=CHECK-OFAST %s
! CHECK-OFAST: -fc1
! CHECK-OFAST-SAME: -ffast-math
! CHECK-OFAST-SAME: -fstack-arrays
! CHECK-OFAST-SAME: -O3

! RUN: %flang -fstack-arrays -fsyntax-only -### %s -o %t 2>&1 \
! RUN:     | FileCheck --check-prefix=CHECK-STACK-ARRAYS %s
! CHECK-STACK-ARRAYS: -fc1
! CHECK-STACK-ARRAYS-SAME: -fstack-arrays

! -Ofast -fno-fast-math => -O3 -fstack-arrays
! RUN: %flang -Ofast -fno-fast-math -fsyntax-only -### %s -o %t 2>&1 \
! RUN:     | FileCheck --check-prefix=CHECK-OFAST-NO-FAST %s
! CHECK-OFAST-NO-FAST: -fc1
! CHECK-OFAST-NO-FAST-NOT: -ffast-math
! CHECK-OFAST-NO-FAST-SAME: -fstack-arrays
! CHECK-OFAST-NO-FAST-SAME: -O3

! -Ofast -fno-stack-arrays -> -O3 -ffast-math
! RUN: %flang -Ofast -fno-stack-arrays -fsyntax-only -### %s -o %t 2>&1 \
! RUN:     | FileCheck --check-prefix=CHECK-OFAST-NO-SA %s
! CHECK-OFAST-NO-SA: -fc1
! CHECK-OFAST-NO-SA-SAME: -ffast-math
! CHECK-OFAST-NO-SA-NOT: -fstack-arrays
! CHECK-OFAST-NO-SA-SAME: -O3

! -ffast-math => -ffast-math
! RUN: %flang -ffast-math -fsyntax-only -### %s -o %t 2>&1 \
! RUN:     | FileCheck --check-prefix=CHECK-FFAST %s
! CHECK-FFAST: -fc1
! CHECK-FFAST-SAME: -ffast-math

! (component flags) => -ffast-math
! RUN: %flang -fsyntax-only -### %s -o %t \
! RUN:     -fno-honor-infinities \
! RUN:     -fno-honor-nans \
! RUN:     -fassociative-math \
! RUN:     -freciprocal-math \
! RUN:     -fapprox-func \
! RUN:     -fno-signed-zeros \
! RUN:     -ffp-contract=fast \
! RUN:     2>&1 | FileCheck --check-prefix=CHECK-FROM-COMPS %s
! CHECK-FROM-COMPS: -fc1
! CHECK-FROM-COMPS-SAME: -ffast-math

! -ffast-math (followed by an alteration) => (component flags)
! RUN: %flang -ffast-math -fhonor-infinities -fsyntax-only -### %s -o %t 2>&1 \
! RUN:     | FileCheck --check-prefix=CHECK-TO-COMPS %s
! CHECK-TO-COMPS: -fc1
! CHECK-TO-COMPS-SAME: -ffp-contract=fast
! CHECK-TO-COMPS-SAME: -menable-no-nans
! CHECK-TO-COMPS-SAME: -fapprox-func
! CHECK-TO-COMPS-SAME: -fno-signed-zeros
! CHECK-TO-COMPS-SAME: -mreassociate
! CHECK-TO-COMPS-SAME: -freciprocal-math

! Check that -fno-fast-math doesn't clobber -ffp-contract
! RUN: %flang -ffp-contract=off -fno-fast-math -fsyntax-only -### %s -o %t 2>&1 \
! RUN:     | FileCheck --check-prefix=CHECK-CONTRACT %s
! CHECK-CONTRACT: -fc1
! CHECK-CONTRACT-SAME: -ffp-contract=off

! Explicit -ffp-contract options participate in normal command-line ordering.
! RUN: %flang -ffast-math -ffp-contract=off -fsyntax-only -### %s -o %t 2>&1 \
! RUN:     | FileCheck --check-prefix=CHECK-FAST-THEN-CONTRACT-OFF \
! RUN:                 --implicit-check-not=-ffast-math %s
! CHECK-FAST-THEN-CONTRACT-OFF: -fc1
! CHECK-FAST-THEN-CONTRACT-OFF-SAME: -ffp-contract=off
! CHECK-FAST-THEN-CONTRACT-OFF-SAME: -menable-no-infs
! CHECK-FAST-THEN-CONTRACT-OFF-SAME: -menable-no-nans
! CHECK-FAST-THEN-CONTRACT-OFF-SAME: -fapprox-func
! CHECK-FAST-THEN-CONTRACT-OFF-SAME: -fno-signed-zeros
! CHECK-FAST-THEN-CONTRACT-OFF-SAME: -mreassociate
! CHECK-FAST-THEN-CONTRACT-OFF-SAME: -freciprocal-math

! RUN: %flang -ffp-contract=off -ffast-math -fsyntax-only -### %s -o %t 2>&1 \
! RUN:     | FileCheck --check-prefix=CHECK-CONTRACT-OFF-THEN-FAST \
! RUN:                 --implicit-check-not=-ffp-contract=off %s
! CHECK-CONTRACT-OFF-THEN-FAST: -fc1
! CHECK-CONTRACT-OFF-THEN-FAST-SAME: -ffast-math

! -Ofast has the same contraction ordering as -ffast-math.
! RUN: %flang -Ofast -ffp-contract=off -fsyntax-only -### %s -o %t 2>&1 \
! RUN:     | FileCheck --check-prefix=CHECK-OFAST-THEN-CONTRACT-OFF %s
! CHECK-OFAST-THEN-CONTRACT-OFF: -fc1
! CHECK-OFAST-THEN-CONTRACT-OFF-SAME: -ffp-contract=off
! CHECK-OFAST-THEN-CONTRACT-OFF-SAME: -menable-no-infs
! CHECK-OFAST-THEN-CONTRACT-OFF-SAME: -menable-no-nans
! CHECK-OFAST-THEN-CONTRACT-OFF-SAME: -fapprox-func
! CHECK-OFAST-THEN-CONTRACT-OFF-SAME: -fno-signed-zeros
! CHECK-OFAST-THEN-CONTRACT-OFF-SAME: -mreassociate
! CHECK-OFAST-THEN-CONTRACT-OFF-SAME: -freciprocal-math
! CHECK-OFAST-THEN-CONTRACT-OFF-SAME: -fstack-arrays
! CHECK-OFAST-THEN-CONTRACT-OFF-SAME: -O3

! Disabling fast math restores the last explicit contraction setting.
! RUN: %flang -ffp-contract=off -ffast-math -fno-fast-math \
! RUN:     -fsyntax-only -### %s -o %t 2>&1 \
! RUN:     | FileCheck --check-prefix=CHECK-RESTORE-CONTRACT-OFF \
! RUN:                 --implicit-check-not=-ffast-math %s
! CHECK-RESTORE-CONTRACT-OFF: -fc1
! CHECK-RESTORE-CONTRACT-OFF-SAME: -ffp-contract=off

! RUN: %flang -ffp-contract=fast -ffast-math -fno-fast-math \
! RUN:     -fsyntax-only -### %s -o %t 2>&1 \
! RUN:     | FileCheck --check-prefix=CHECK-RESTORE-CONTRACT-FAST \
! RUN:                 --implicit-check-not=-ffast-math %s
! CHECK-RESTORE-CONTRACT-FAST: -fc1
! CHECK-RESTORE-CONTRACT-FAST-SAME: -ffp-contract=fast

! Check that -ffast-math causes us to link to crtfastmath.o
! UNSUPPORTED: system-windows
! UNSUPPORTED: target=powerpc{{.*}}
! RUN: %flang -ffast-math -### %s -o %t 2>&1 \
! RUN:           --target=x86_64-unknown-linux -no-pie \
! RUN:           --sysroot=%S/../../../clang/test/Driver/Inputs/basic_linux_tree \
! RUN:     | FileCheck --check-prefix=CHECK-CRT %s
! CHECK-CRT: {{crtbegin.?\.o}}
! CHECK-CRT-SAME: crtfastmath.o
! CHECK-CRT-SAME: {{crtend.?\.o}}

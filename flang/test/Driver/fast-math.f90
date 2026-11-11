! Test for correct forwarding of fast-math flags from the compiler driver to the
! frontend driver

! Check warning message for Ofast deprecation
! RUN: %flang -Ofast -### %s 2>&1 | FileCheck %s
! CHECK: warning: argument '-Ofast' is deprecated; use '-O3 -ffast-math -fstack-arrays -fno-protect-parens' for the same behavior, or '-O3 -fstack-arrays' to enable only conforming optimizations [-Wdeprecated-ofast]

! -Ofast => -ffast-math -O3 -fstack-arrays
! RUN: %flang -Ofast -fsyntax-only -### %s 2>&1 \
! RUN:     | FileCheck --check-prefix=CHECK-OFAST %s
! CHECK-OFAST: -fc1
! CHECK-OFAST-SAME: -ffast-math
! CHECK-OFAST-SAME: -fstack-arrays
! CHECK-OFAST-SAME: -O3

! RUN: %flang -fstack-arrays -fsyntax-only -### %s 2>&1 \
! RUN:     | FileCheck --check-prefix=CHECK-STACK-ARRAYS %s
! CHECK-STACK-ARRAYS: -fc1
! CHECK-STACK-ARRAYS-SAME: -fstack-arrays

! -Ofast -fno-fast-math => -O3 -fstack-arrays
! RUN: %flang -Ofast -fno-fast-math -fsyntax-only -### %s 2>&1 \
! RUN:     | FileCheck --check-prefix=CHECK-OFAST-NO-FAST %s
! CHECK-OFAST-NO-FAST: -fc1
! CHECK-OFAST-NO-FAST-NOT: -ffast-math
! CHECK-OFAST-NO-FAST-SAME: -fstack-arrays
! CHECK-OFAST-NO-FAST-SAME: -O3

! -Ofast -fno-stack-arrays -> -O3 -ffast-math
! RUN: %flang -Ofast -fno-stack-arrays -fsyntax-only -### %s 2>&1 \
! RUN:     | FileCheck --check-prefix=CHECK-OFAST-NO-SA %s
! CHECK-OFAST-NO-SA: -fc1
! CHECK-OFAST-NO-SA-SAME: -ffast-math
! CHECK-OFAST-NO-SA-NOT: -fstack-arrays
! CHECK-OFAST-NO-SA-SAME: -O3

! -ffast-math => -ffast-math
! RUN: %flang -ffast-math -fsyntax-only -### %s 2>&1 \
! RUN:     | FileCheck --check-prefix=CHECK-FFAST %s
! CHECK-FFAST: -fc1
! CHECK-FFAST-SAME: -ffast-math

! (component flags) => -ffast-math
! RUN: %flang -fsyntax-only -### %s \
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
! RUN: %flang -ffast-math -fhonor-infinities -fsyntax-only -### %s 2>&1 \
! RUN:     | FileCheck --check-prefix=CHECK-TO-COMPS %s
! CHECK-TO-COMPS: -fc1
! CHECK-TO-COMPS-SAME: -ffp-contract=fast
! CHECK-TO-COMPS-SAME: -menable-no-nans
! CHECK-TO-COMPS-SAME: -fapprox-func
! CHECK-TO-COMPS-SAME: -fno-signed-zeros
! CHECK-TO-COMPS-SAME: -mreassociate
! CHECK-TO-COMPS-SAME: -freciprocal-math

! Check if -ffast-math component flags can be disabled
! RUN: %flang -ffast-math \
! RUN:      -ffp-contract=off \
! RUN:      -fhonor-infinities \
! RUN:      -fhonor-nans \
! RUN:      -fno-approx-func \
! RUN:      -fsigned-zeros \
! RUN:      -fno-associative-math \
! RUN:      -fno-reciprocal-math \
! RUN:      -fsyntax-only -### %s 2>&1 \
! RUN:      | FileCheck --check-prefix=CHECK-TO-COMPS-DIS %s
! CHECK-TO-COMPS-DIS: warning: overriding '-ffast-math' option with '-ffp-contract=off' [-Woverriding-option]
! CHECK-TO-COMPS-DIS: -fc1
! CHECK-TO-COMPS-DIS-SAME: -ffp-contract=off
! CHECK-TO-COMPS-DIS-NOT: -menable-no-infs
! CHECK-TO-COMPS-DIS-NOT: -menable-no-nans
! CHECK-TO-COMPS-DIS-NOT: -fapprox-func
! CHECK-TO-COMPS-DIS-NOT: -fno-signed-zeros
! CHECK-TO-COMPS-DIS-NOT: -mreassociate
! CHECK-TO-COMPS-DIS-NOT: -freciprocal-math

! Check that -fno-fast-math doesn't clobber -ffp-contract
! RUN: %flang -ffp-contract=off -fno-fast-math -fsyntax-only -### %s 2>&1 \
! RUN:     | FileCheck --check-prefix=CHECK-CONTRACT %s
! CHECK-CONTRACT: -fc1
! CHECK-CONTRACT-SAME: -ffp-contract=off

! Check that -fno-fast-math only disables -ffast-math.
! RUN: %flang -ffp-contract=off -ffast-math -fno-fast-math -fsyntax-only -### %s 2>&1 \
! RUN:     | FileCheck --check-prefix=CHECK-CONTRACT-NOFAST %s
! CHECK-CONTRACT-NOFAST: -fc1
! CHECK-CONTRACT-NOFAST-SAME: -ffp-contract=off

! Check that no warning is emitted after -fno-fast-math.
! RUN: %flang -ffast-math -fno-fast-math -ffp-contract=off -fsyntax-only -### %s 2>&1 \
! RUN:     | FileCheck --check-prefix=CHECK-NOFAST-CONTRACT %s
! CHECK-NOFAST-CONTRACT-NOT: warning: overriding '-ffast-math' option with '-ffp-contract=off' [-Woverriding-option]
! CHECK-NOFAST-CONTRACT: -fc1
! CHECK-NOFAST-CONTRACT-SAME: -ffp-contract=off

! Check that -Ofast is overridden.
! RUN: %flang -Ofast -ffp-contract=off -fsyntax-only -### %s 2>&1 \
! RUN:     | FileCheck --check-prefix=CHECK-OFAST-CONTRACT %s
! CHECK-OFAST-CONTRACT: warning: overriding '-Ofast' option with '-ffp-contract=off' [-Woverriding-option]
! CHECK-OFAST-CONTRACT: -fc1
! CHECK-OFAST-CONTRACT-SAME: -ffp-contract=off

! Check that -ffast-math causes us to link to crtfastmath.o
! UNSUPPORTED: system-windows
! UNSUPPORTED: target=powerpc{{.*}}
! RUN: %flang -ffast-math -### %s 2>&1 \
! RUN:           --target=x86_64-unknown-linux -no-pie \
! RUN:           --sysroot=%S/../../../clang/test/Driver/Inputs/basic_linux_tree \
! RUN:     | FileCheck --check-prefix=CHECK-CRT %s
! CHECK-CRT: {{crtbegin.?\.o}}
! CHECK-CRT-SAME: crtfastmath.o
! CHECK-CRT-SAME: {{crtend.?\.o}}

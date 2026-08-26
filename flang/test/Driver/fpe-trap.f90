! Test the -ffpe-trap= option: driver forwarding and driver-level validation
! (value checking and target-support warnings).

!--- The driver forwards -ffpe-trap= to the frontend ---------------------------

! Test all supported exception types are forwarded to -fc1.
! RUN: %flang -ffpe-trap=invalid,zero,overflow,underflow,inexact,denormal -### %s 2>&1 \
! RUN:     | FileCheck --check-prefix=CHECK-ALL %s
! CHECK-ALL: -fc1
! CHECK-ALL-SAME: -ffpe-trap=invalid,zero,overflow,underflow,inexact,denormal

! Test a single exception type.
! RUN: %flang -ffpe-trap=invalid -### %s 2>&1 \
! RUN:     | FileCheck --check-prefix=CHECK-SINGLE %s
! CHECK-SINGLE: -fc1
! CHECK-SINGLE-SAME: -ffpe-trap=invalid

! Only the last -ffpe-trap= is forwarded to the frontend.
! RUN: %flang -ffpe-trap=invalid -ffpe-trap=overflow -### %s 2>&1 \
! RUN:     | FileCheck --check-prefix=CHECK-LAST %s
! CHECK-LAST: -fc1
! CHECK-LAST-SAME: -ffpe-trap=overflow
! CHECK-LAST-NOT: -ffpe-trap=invalid

! By default (no -ffpe-trap=), nothing is forwarded to the frontend.
! RUN: %flang -### %s 2>&1 | FileCheck --check-prefix=CHECK-DEFAULT %s
! CHECK-DEFAULT: -fc1
! CHECK-DEFAULT-NOT: -ffpe-trap

!--- "none" and an empty list are accepted -------------------------------------

! These are valid and must not produce a driver error (a non-zero exit code
! would make these RUN lines fail).
! RUN: %flang -ffpe-trap=none -### %s
! RUN: %flang -ffpe-trap= -### %s
! RUN: %flang -ffpe-trap=invalid,none -### %s

!--- The driver rejects an unknown mnemonic ------------------------------------

! RUN: not %flang -ffpe-trap=bogus -### %s 2>&1 \
! RUN:     | FileCheck --check-prefix=CHECK-BADARG %s
! CHECK-BADARG: error: unsupported argument 'bogus' to option '-ffpe-trap='

!--- -ffpe-trap= is independent of -ffast-math / -Ofast ------------------------

! -ffpe-trap= is not part of the fast-math option set, so it is still forwarded
! and still validated when -ffast-math/-Ofast is present.
! RUN: %flang -Ofast -ffpe-trap=invalid -### %s 2>&1 \
! RUN:     | FileCheck --check-prefix=CHECK-FASTMATH %s
! CHECK-FASTMATH: -fc1
! CHECK-FASTMATH-SAME: -ffpe-trap=invalid

! RUN: not %flang -Ofast -ffpe-trap=bogus -### %s 2>&1 \
! RUN:     | FileCheck --check-prefix=CHECK-BADARG %s

!--- The driver warns when the target cannot honor the request -----------------

! On a target without floating-point halting support, the driver warns and the
! option is ignored at run time.
! RUN: %flang --target=powerpc64-ibm-aix -ffpe-trap=invalid -### %s 2>&1 \
! RUN:     | FileCheck --check-prefix=CHECK-WARN %s
! CHECK-WARN: warning: ignoring '-ffpe-trap=invalid' option as it is not currently supported for target 'powerpc64-ibm-aix'

! On a supported non-x86 target (glibc/Linux), no warning is emitted.
! RUN: %flang --target=aarch64-unknown-linux-gnu -ffpe-trap=invalid -### %s 2>&1 \
! RUN:     | FileCheck --check-prefix=CHECK-NOWARN --allow-empty %s
! CHECK-NOWARN-NOT: ignoring '-ffpe-trap

! The "denormal" exception is an x86-only extension: requesting it for a non-x86
! target warns even though the target is otherwise supported.
! RUN: %flang --target=aarch64-unknown-linux-gnu -ffpe-trap=invalid,denormal -### %s 2>&1 \
! RUN:     | FileCheck --check-prefix=CHECK-DENORM %s
! CHECK-DENORM: warning: ignoring '-ffpe-trap=denormal' option as it is not currently supported for target 'aarch64-unknown-linux-gnu'

! On x86 the "denormal" exception is supported, so no warning is emitted.
! RUN: %flang --target=x86_64-unknown-linux-gnu -ffpe-trap=denormal -### %s 2>&1 \
! RUN:     | FileCheck --check-prefix=CHECK-X86DENORM --allow-empty %s
! CHECK-X86DENORM-NOT: ignoring '-ffpe-trap

end program

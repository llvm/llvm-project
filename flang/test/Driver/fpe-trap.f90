! Test that -ffpe-trap= is forwarded from the Flang driver to the frontend.

! RUN: %flang -ffpe-trap=invalid,zero,overflow -fsyntax-only -### %s 2>&1 \
! RUN:     | FileCheck --check-prefix=CHECK-FORWARD %s
! CHECK-FORWARD: -fc1
! CHECK-FORWARD-SAME: -ffpe-trap=invalid,zero,overflow

! Test all supported exception types
! RUN: %flang -ffpe-trap=invalid,zero,overflow,underflow,inexact,denormal -fsyntax-only -### %s 2>&1 \
! RUN:     | FileCheck --check-prefix=CHECK-ALL %s
! CHECK-ALL: -fc1
! CHECK-ALL-SAME: -ffpe-trap=invalid,zero,overflow,underflow,inexact,denormal

! Test a single exception type
! RUN: %flang -ffpe-trap=invalid -fsyntax-only -### %s 2>&1 \
! RUN:     | FileCheck --check-prefix=CHECK-SINGLE %s
! CHECK-SINGLE: -fc1
! CHECK-SINGLE-SAME: -ffpe-trap=invalid

! Only the last -ffpe-trap= is forwarded to the frontend.
! RUN: %flang -ffpe-trap=invalid -ffpe-trap=overflow -fsyntax-only -### %s 2>&1 \
! RUN:     | FileCheck --check-prefix=CHECK-LAST %s
! CHECK-LAST: -fc1
! CHECK-LAST-SAME: -ffpe-trap=overflow
! CHECK-LAST-NOT: -ffpe-trap=invalid

! "none" and an empty list are accepted and disable halting.
! RUN: %flang_fc1 -ffpe-trap=none -fsyntax-only %s 2>&1 | FileCheck --check-prefix=CHECK-OK --allow-empty %s
! RUN: %flang_fc1 -ffpe-trap= -fsyntax-only %s 2>&1 | FileCheck --check-prefix=CHECK-OK --allow-empty %s
! RUN: %flang_fc1 -ffpe-trap=invalid,none -fsyntax-only %s 2>&1 | FileCheck --check-prefix=CHECK-OK --allow-empty %s
! CHECK-OK-NOT: error:

! On a target without floating-point halting support, the driver warns and still
! forwards the option (the runtime ignores it at execution time).
! RUN: %flang --target=powerpc64-ibm-aix -ffpe-trap=invalid -fsyntax-only -### %s 2>&1 \
! RUN:     | FileCheck --check-prefix=CHECK-WARN %s
! CHECK-WARN: warning: ignoring '-ffpe-trap=invalid' option as it is not currently supported for target 'powerpc64-ibm-aix'
! CHECK-WARN: -ffpe-trap=invalid

! On a supported non-x86 target (glibc/Linux), no warning is emitted.
! RUN: %flang --target=aarch64-unknown-linux-gnu -ffpe-trap=invalid -fsyntax-only -### %s 2>&1 \
! RUN:     | FileCheck --check-prefix=CHECK-NOWARN %s
! CHECK-NOWARN-NOT: ignoring '-ffpe-trap
! CHECK-NOWARN: -ffpe-trap=invalid

! The "denormal" exception is an x86-only extension: requesting it for a non-x86
! target warns even though the target is otherwise supported. The rest of the
! list is still honored, so the option is still forwarded.
! RUN: %flang --target=aarch64-unknown-linux-gnu -ffpe-trap=invalid,denormal -fsyntax-only -### %s 2>&1 \
! RUN:     | FileCheck --check-prefix=CHECK-DENORM %s
! CHECK-DENORM: warning: ignoring '-ffpe-trap=denormal' option as it is not currently supported for target 'aarch64-unknown-linux-gnu'
! CHECK-DENORM: -ffpe-trap=invalid,denormal

! On x86 the "denormal" exception is supported, so no warning is emitted.
! RUN: %flang --target=x86_64-unknown-linux-gnu -ffpe-trap=denormal -fsyntax-only -### %s 2>&1 \
! RUN:     | FileCheck --check-prefix=CHECK-X86DENORM %s
! CHECK-X86DENORM-NOT: ignoring '-ffpe-trap
! CHECK-X86DENORM: -ffpe-trap=denormal

! Test invalid exception type
! RUN: not %flang_fc1 -ffpe-trap=bogus -fsyntax-only %s 2>&1 \
! RUN:     | FileCheck --check-prefix=CHECK-INVALID %s
! CHECK-INVALID: error: unsupported argument 'bogus' to option '-ffpe-trap='

end program

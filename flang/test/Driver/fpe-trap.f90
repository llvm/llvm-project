! Test the -ffpe-trap= option: driver forwarding, and frontend validation and
! target-support warnings.

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

!--- The frontend accepts "none" and an empty list -----------------------------

! "none" and an empty list are accepted and disable halting. A parse error would
! make these fail with a non-zero exit code.
! RUN: %flang_fc1 -ffpe-trap=none -fsyntax-only %s
! RUN: %flang_fc1 -ffpe-trap= -fsyntax-only %s
! RUN: %flang_fc1 -ffpe-trap=invalid,none -fsyntax-only %s

!--- The frontend rejects an unknown mnemonic ----------------------------------

! RUN: not %flang_fc1 -ffpe-trap=bogus -fsyntax-only %s 2>&1 \
! RUN:     | FileCheck --check-prefix=CHECK-BADARG %s
! CHECK-BADARG: error: unsupported argument 'bogus' to option '-ffpe-trap='

!--- The frontend warns when the target cannot honor the request ---------------
!
! These runs invoke -fc1 with a specific -triple, so they are guarded by the
! matching *-registered-target feature (the frontend creates a target machine
! even for -fsyntax-only).

! On a target without floating-point halting support, the frontend warns and the
! option is ignored at run time.
! RUN: %if powerpc-registered-target %{ \
! RUN:     %flang_fc1 -triple powerpc64-ibm-aix -ffpe-trap=invalid -fsyntax-only %s 2>&1 \
! RUN:     | FileCheck --check-prefix=CHECK-WARN %s %}
! CHECK-WARN: warning: ignoring '-ffpe-trap=invalid' option as it is not currently supported for target 'powerpc64-ibm-aix'

! On a supported non-x86 target (glibc/Linux), no warning is emitted.
! RUN: %if aarch64-registered-target %{ \
! RUN:     %flang_fc1 -triple aarch64-unknown-linux-gnu -ffpe-trap=invalid -fsyntax-only %s 2>&1 \
! RUN:     | FileCheck --check-prefix=CHECK-NOWARN --allow-empty %s %}
! CHECK-NOWARN-NOT: ignoring '-ffpe-trap

! The "denormal" exception is an x86-only extension: requesting it for a non-x86
! target warns even though the target is otherwise supported.
! RUN: %if aarch64-registered-target %{ \
! RUN:     %flang_fc1 -triple aarch64-unknown-linux-gnu -ffpe-trap=invalid,denormal -fsyntax-only %s 2>&1 \
! RUN:     | FileCheck --check-prefix=CHECK-DENORM %s %}
! CHECK-DENORM: warning: ignoring '-ffpe-trap=denormal' option as it is not currently supported for target 'aarch64-unknown-linux-gnu'

! On x86 the "denormal" exception is supported, so no warning is emitted.
! RUN: %if x86-registered-target %{ \
! RUN:     %flang_fc1 -triple x86_64-unknown-linux-gnu -ffpe-trap=denormal -fsyntax-only %s 2>&1 \
! RUN:     | FileCheck --check-prefix=CHECK-X86DENORM --allow-empty %s %}
! CHECK-X86DENORM-NOT: ignoring '-ffpe-trap

end program

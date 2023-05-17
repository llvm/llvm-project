! Test that -print-target-triple prints correct triple

! RUN: %flang -print-target-triple 2>&1 --target=aarch64-linux-gnu \
! RUN:   | FileCheck --check-prefix=AARCH64 %s

! RUN: %flang -print-target-triple 2>&1 --target=x86_64-linux-gnu \
! RUN:   | FileCheck --check-prefix=X86_64 %s

! X86_64: x86_64-unknown-linux-gnu
! AARCH64: aarch64-unknown-linux-gnu

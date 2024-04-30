! Test that -dumpmachine prints the target triple.

! Note: Debian GCC may omit "unknown-".
! RUN: %flang --target=x86_64-linux-gnu -dumpmachine | FileCheck %s --check-prefix=X86_64
! X86_64: x86_64-unknown-linux-gnu

! RUN: %flang --target=xxx-pc-freebsd -dumpmachine | FileCheck %s --check-prefix=FREEBSD
! FREEBSD: xxx-pc-freebsd

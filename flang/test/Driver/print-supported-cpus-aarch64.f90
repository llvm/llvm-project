! Test --print-supported-cpus and associated aliases, -mcpu=help and
! -mtune=help on AArch64.

! REQUIRES: aarch64-registered-target

! RUN: %flang --target=aarch64-unknown-linux-gnu --print-supported-cpus 2>&1 | \
! RUN:   FileCheck %s
! RUN: %flang --target=aarch64-unknown-linux-gnu -mcpu=help 2>&1 | \
! RUN:   FileCheck %s
! RUN: %flang --target=aarch64-unknown-linux-gnu -mtune=help 2>&1 | \
! RUN:   FileCheck %s

! CHECK-NOT: warning: argument unused during compilation

! CHECK: Target: aarch64-unknown-linux-gnu
! CHECK: cortex-a73
! CHECK: cortex-a75

! TODO: This is a line that is printed at the end of the output. The full line
! also includes an example that references clang. That needs to be fixed and a
! a check added here to make sure that it references flang, not clang.

! CHECK: Use -mcpu or -mtune to specify the target's processor.

! Test --print-supported-cpus and associated aliases, -mcpu=help and
! -mtune=help

! RUN: %if x86-registered-target %{ \
! RUN:   %flang --target=x86_64-unknown-linux-gnu --print-supported-cpus 2>&1 \
! RUN:     | FileCheck %s --check-prefixes=X86,CHECK \
! RUN: %}
! RUN: %if x86-registered-target %{ \
! RUN:   %flang --target=x86_64-unknown-linux-gnu -mcpu=help 2>&1 \
! RUN:     | FileCheck %s --check-prefixes=X86,CHECK \
! RUN: %}
! RUN: %if x86-registered-target %{ \
! RUN:   %flang --target=x86_64-unknown-linux-gnu -mtune=help 2>&1 \
! RUN:     | FileCheck %s --check-prefixes=X86,CHECK \
! RUN: %}

! RUN: %if aarch64-registered-target %{ \
! RUN:   %flang --target=aarch64-unknown-linux-gnu --print-supported-cpus 2>&1 \
! RUN:     | FileCheck %s --check-prefixes=AARCH64,CHECK \
! RUN: %}
! RUN: %if x86-registered-target %{ \
! RUN:   %flang --target=aarch64-unknown-linux-gnu -mcpu=help 2>&1 \
! RUN:     | FileCheck %s --check-prefixes=AARCH64,CHECK \
! RUN: %}
! RUN: %if x86-registered-target %{ \
! RUN:   %flang --target=aarch64-unknown-linux-gnu -mtune=help 2>&1 \
! RUN:     | FileCheck %s --check-prefixes=AARCH64,CHECK \
! RUN: %}

! CHECK-NOT: warning: argument unused during compilation

! X86: Target: x86_64-unknown-linux-gnu
! X86: corei7

! AARCH64: Target: aarch64-unknown-linux-gnu
! AARCH64: cortex-a73
! AARCH64: cortex-a75

! TODO: This is a line that is printed at the end of the output. The full line
! also includes an example that references clang. That needs to be fixed and a
! a check added here to make sure that it references flang, not clang.

! CHECK: Use -mcpu or -mtune to specify the target's processor.

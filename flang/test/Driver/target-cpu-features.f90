! REQUIRES: aarch64-registered-target, x86-registered-target

! Test that -mcpu/march are used and that the -target-cpu and -target-features
! are also added to the fc1 command.

! RUN: %flang --target=aarch64-linux-gnu -mcpu=cortex-a57 -c %s -### 2>&1 \
! RUN: | FileCheck %s -check-prefix=CHECK-A57

! RUN: %flang --target=aarch64-linux-gnu -mcpu=cortex-a76 -c %s -### 2>&1 \
! RUN: | FileCheck %s -check-prefix=CHECK-A76

! RUN: %flang --target=aarch64-linux-gnu -march=armv9 -c %s -### 2>&1 \
! RUN: | FileCheck %s -check-prefix=CHECK-ARMV9

! Negative test. ARM cpu with x86 target.
! RUN: %flang --target=x86_64-linux-gnu -mcpu=cortex-a57 -c %s -### 2>&1 \
! RUN: | FileCheck %s -check-prefix=CHECK-NO-A57

! RUN: %flang --target=x86_64-linux-gnu -march=skylake -c %s -### 2>&1 \
! RUN: | FileCheck %s -check-prefix=CHECK-SKYLAKE

! RUN: %flang --target=x86_64h-linux-gnu -c %s -### 2>&1 \
! RUN: | FileCheck %s -check-prefix=CHECK-X86_64H


! Test that invalid cpu and features are ignored.

! RUN: %flang_fc1 -triple aarch64-linux-gnu -target-cpu supercpu \
! RUN: -o /dev/null -S %s 2>&1 | FileCheck %s -check-prefix=CHECK-INVALID-CPU

! RUN: %flang_fc1 -triple aarch64-linux-gnu -target-feature +superspeed \
! RUN: -o /dev/null -S %s 2>&1 | FileCheck %s -check-prefix=CHECK-INVALID-FEATURE


! CHECK-A57: "-fc1" "-triple" "aarch64-unknown-linux-gnu"
! CHECK-A57-SAME: "-target-cpu" "cortex-a57" "-target-feature" "+v8a" "-target-feature" "+aes" "-target-feature" "+crc" "-target-feature" "+fp-armv8" "-target-feature" "+sha2" "-target-feature" "+neon"

! CHECK-A76: "-fc1" "-triple" "aarch64-unknown-linux-gnu"
! CHECK-A76-SAME: "-target-cpu" "cortex-a76" "-target-feature" "+v8.2a" "-target-feature" "+aes" "-target-feature" "+crc" "-target-feature" "+dotprod" "-target-feature" "+fp-armv8" "-target-feature" "+fullfp16" "-target-feature" "+lse" "-target-feature" "+ras" "-target-feature" "+rcpc" "-target-feature" "+rdm" "-target-feature" "+sha2" "-target-feature" "+neon" "-target-feature" "+ssbs"

! CHECK-ARMV9: "-fc1" "-triple" "aarch64-unknown-linux-gnu"
! CHECK-ARMV9-SAME: "-target-cpu" "generic" "-target-feature" "+neon" "-target-feature" "+v9a" "-target-feature" "+sve" "-target-feature" "+sve2"

! CHECK-NO-A57: "-fc1" "-triple" "x86_64-unknown-linux-gnu"
! CHECK-NO-A57-NOT: cortex-a57
! CHECK-NO-A57-SAME: "-target-cpu" "x86-64"
! CHECK-NO-A57-NOT: cortex-a57

! CHECK-SKYLAKE: "-fc1" "-triple" "x86_64-unknown-linux-gnu"
! CHECK-SKYLAKE-SAME: "-target-cpu" "skylake"

! CHECK-X86_64H: "-fc1" "-triple" "x86_64h-unknown-linux-gnu"
! CHECK-X86_64H-SAME: "-target-cpu" "x86-64" "-target-feature" "-rdrnd" "-target-feature" "-aes" "-target-feature" "-pclmul" "-target-feature" "-rtm" "-target-feature" "-fsgsbase"

! CHECK-INVALID-CPU: 'supercpu' is not a recognized processor for this target (ignoring processor)
! CHECK-INVALID-FEATURE: '+superspeed' is not a recognized feature for this target (ignoring feature)

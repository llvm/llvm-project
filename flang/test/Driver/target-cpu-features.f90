! Test that -mcpu/march are used and that the -target-cpu and -target-features
! are also added to the fc1 command.

! RUN: %flang --target=aarch64-linux-gnu -mcpu=cortex-a57 -c %s -### 2>&1 \
! RUN: | FileCheck %s -check-prefix=CHECK-A57

! RUN: %flang --target=aarch64-linux-gnu -mcpu=cortex-a76 -c %s -### 2>&1 \
! RUN: | FileCheck %s -check-prefix=CHECK-A76

! RUN: %flang --target=aarch64-linux-gnu -march=armv9 -c %s -### 2>&1 \
! RUN: | FileCheck %s -check-prefix=CHECK-ARMV9

! Negative test. ARM cpu with x86 target.
! RUN: not %flang --target=x86_64-linux-gnu -mcpu=cortex-a57 -c %s -### 2>&1 \
! RUN: | FileCheck %s -check-prefix=CHECK-NO-A57

! RUN: %flang --target=x86_64-linux-gnu -march=skylake -c %s -### 2>&1 \
! RUN: | FileCheck %s -check-prefix=CHECK-SKYLAKE

! RUN: %flang --target=x86_64-linux-gnu -mapx-features=egpr -c %s -### 2>&1 \
! RUN: | FileCheck %s -check-prefix=CHECK-APX

! RUN: %flang --target=x86_64-linux-gnu -mno-apx-features=ccmp -c %s -### 2>&1 \
! RUN: | FileCheck %s -check-prefix=CHECK-NO-APX

! RUN: %flang --target=x86_64-linux-gnu -mevex512 -c %s -### 2>&1 \
! RUN: | FileCheck %s -check-prefix=CHECK-EVEX512

! RUN: %flang --target=x86_64-linux-gnu -mno-evex512 -c %s -### 2>&1 \
! RUN: | FileCheck %s -check-prefix=CHECK-NO-EVEX512

! RUN: %flang --target=x86_64h-linux-gnu -c %s -### 2>&1 \
! RUN: | FileCheck %s -check-prefix=CHECK-X86_64H

! RUN: %flang --target=riscv64-linux-gnu -c %s -### 2>&1 \
! RUN: | FileCheck %s -check-prefix=CHECK-RV64

! RUN: %flang --target=amdgcn-amd-amdhsa -mcpu=gfx908 -nogpulib -c %s -### 2>&1 \
! RUN: | FileCheck %s -check-prefix=CHECK-AMDGPU

! RUN: %flang --target=r600-unknown-unknown -mcpu=cayman -nogpulib -c %s -### 2>&1 \
! RUN: | FileCheck %s -check-prefix=CHECK-AMDGPU-R600

! RUN: %flang --target=loongarch64-linux-gnu -c %s -### 2>&1 \
! RUN: | FileCheck %s -check-prefix=CHECK-LOONGARCH64

! RUN: %flang --target=sparc64-linux-gnu -c -### %s 2>&1  | FileCheck %s -check-prefix=CHECK-SPARC-VIS
! RUN: %flang --target=sparc64-freebsd -c -### %s 2>&1  | FileCheck %s -check-prefix=CHECK-SPARC-VIS
! RUN: %flang --target=sparc64-openbsd -c -### %s 2>&1  | FileCheck %s -check-prefix=CHECK-SPARC-VIS

! CHECK-A57: "-fc1" "-triple" "aarch64-unknown-linux-gnu"
! CHECK-A57-SAME: "-target-cpu" "cortex-a57"
! CHECK-A57-SAME: "-target-feature" "+v8a" "-target-feature" "+aes" "-target-feature" "+crc" "-target-feature" "+fp-armv8" "-target-feature" "+neon" "-target-feature" "+perfmon" "-target-feature" "+sha2

! CHECK-A76: "-fc1" "-triple" "aarch64-unknown-linux-gnu"
! CHECK-A76-SAME: "-target-cpu" "cortex-a76"
! CHECK-A76-SAME: "-target-feature" "+v8.2a" "-target-feature" "+aes" "-target-feature" "+crc" "-target-feature" "+dotprod" "-target-feature" "+fp-armv8" "-target-feature" "+fullfp16" "-target-feature" "+lse" "-target-feature" "+neon" "-target-feature" "+perfmon" "-target-feature" "+ras" "-target-feature" "+rcpc" "-target-feature" "+rdm" "-target-feature" "+sha2" "-target-feature" "+ssbs"

! CHECK-ARMV9: "-fc1" "-triple" "aarch64-unknown-linux-gnu"
! CHECK-ARMV9-SAME: "-target-cpu" "generic"
! CHECK-ARMV9-SAME: "-target-feature" "+v9a"
! CHECK-ARMV9-SAME: "-target-feature" "+sve"
! CHECK-ARMV9-SAME: "-target-feature" "+sve2"

! CHECK-NO-A57: "-fc1" "-triple" "x86_64-unknown-linux-gnu"
! CHECK-NO-A57-NOT: cortex-a57
! CHECK-NO-A57-SAME: "-target-cpu" "x86-64"
! CHECK-NO-A57-NOT: cortex-a57

! CHECK-SKYLAKE: "-fc1" "-triple" "x86_64-unknown-linux-gnu"
! CHECK-SKYLAKE-SAME: "-target-cpu" "skylake"

! CHECK-APX: "-fc1" "-triple" "x86_64-unknown-linux-gnu"
! CHECK-APX-SAME: "-target-feature" "+egpr"

! CHECK-NO-APX: "-fc1" "-triple" "x86_64-unknown-linux-gnu"
! CHECK-NO-APX-SAME: "-target-feature" "-ccmp"

! CHECK-EVEX512: "-fc1" "-triple" "x86_64-unknown-linux-gnu"
! CHECK-EVEX512-SAME: "-target-feature" "+evex512"

! CHECK-NO-EVEX512: "-fc1" "-triple" "x86_64-unknown-linux-gnu"
! CHECK-NO-EVEX512-SAME: "-target-feature" "-evex512"

! CHECK-X86_64H: "-fc1" "-triple" "x86_64h-unknown-linux-gnu"
! CHECK-X86_64H-SAME: "-target-cpu" "x86-64" "-target-feature" "-rdrnd" "-target-feature" "-aes" "-target-feature" "-pclmul" "-target-feature" "-rtm" "-target-feature" "-fsgsbase"

! CHECK-RV64: "-fc1" "-triple" "riscv64-unknown-linux-gnu"
! CHECK-RV64-SAME: "-target-cpu" "generic-rv64" "-target-feature" "+m" "-target-feature" "+a" "-target-feature" "+f" "-target-feature" "+d" "-target-feature" "+c"

! CHECK-AMDGPU: "-fc1" "-triple" "amdgcn-amd-amdhsa"
! CHECK-AMDGPU-SAME: "-target-cpu" "gfx908"

! CHECK-AMDGPU-R600: "-fc1" "-triple" "r600-unknown-unknown"
! CHECK-AMDGPU-R600-SAME: "-target-cpu" "cayman"

! CHECK-LOONGARCH64: "-fc1" "-triple" "loongarch64-unknown-linux-gnu"
! CHECK-LOONGARCH64-SAME: "-target-cpu" "loongarch64" "-target-feature" "+lsx" "-target-feature" "+64bit" "-target-feature" "+f" "-target-feature" "+d" "-target-feature" "+ual"

! CHECK-SPARC-VIS: "-fc1" "-triple" "sparc64-{{[^"]+}}"
! CHECK-SPARC-VIS-SAME: "-target-feature" "+vis"

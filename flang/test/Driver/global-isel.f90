! RUN: %flang -fglobal-isel -S -### %s 2>&1 | FileCheck --check-prefix=ENABLED %s
! RUN: %flang -fno-global-isel -S -### %s 2>&1 | FileCheck --check-prefix=DISABLED %s

! RUN: %flang -target aarch64 -fglobal-isel -S %s -### 2>&1 | FileCheck --check-prefix=ARM64-DEFAULT %s
! RUN: %flang -target aarch64 -fglobal-isel -S -O0 %s -### 2>&1 | FileCheck --check-prefix=ARM64-O0 %s
! RUN: %flang -target aarch64 -fglobal-isel -S -O2 %s -### 2>&1 | FileCheck --check-prefix=ARM64-O2 %s

! RUN: %flang -target x86_64 -fglobal-isel -S %s -### 2>&1 | FileCheck --check-prefix=X86_64 %s

! Now test the aliases.

! RUN: %flang -fexperimental-isel -S -### %s 2>&1 | FileCheck --check-prefix=ENABLED %s
! RUN: %flang -fno-experimental-isel -S -### %s 2>&1 | FileCheck --check-prefix=DISABLED %s

! RUN: %flang -target aarch64 -fexperimental-isel -S %s -### 2>&1 | FileCheck --check-prefix=ARM64-DEFAULT %s
! RUN: %flang -target aarch64 -fexperimental-isel -S -O0 %s -### 2>&1 | FileCheck --check-prefix=ARM64-O0 %s
! RUN: %flang -target aarch64 -fexperimental-isel -S -O2 %s -### 2>&1 | FileCheck --check-prefix=ARM64-O2 %s

! RUN: %flang -target x86_64 -fexperimental-isel -S %s -### 2>&1 | FileCheck --check-prefix=X86_64 %s

! ENABLED: "-mllvm" "-global-isel=1"
! DISABLED: "-mllvm" "-global-isel=0"

! ARM64-DEFAULT-NOT: warning: -fglobal-isel
! ARM64-DEFAULT-NOT: "-global-isel-abort=2"
! ARM64-O0-NOT: warning: -fglobal-isel
! ARM64-O2: warning: -fglobal-isel support is incomplete for this architecture at the current optimization level
! ARM64-O2: "-mllvm" "-global-isel-abort=2"

! X86_64: -fglobal-isel support for the 'x86_64' architecture is incomplete
! X86_64: "-mllvm" "-global-isel-abort=2"

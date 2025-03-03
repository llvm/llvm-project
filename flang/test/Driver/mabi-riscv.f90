! RUN: not %flang -c --target=riscv64-unknown-linux -mabi=ilp32 %s -### 2>&1 | FileCheck --check-prefix=INVALID1 %s
! RUN: not %flang -c --target=riscv64-unknown-linux -mabi=lp64e %s -### 2>&1 | FileCheck --check-prefix=INVALID2 %s
! RUN: %flang -c --target=riscv64-unknown-linux -mabi=lp64 %s -### 2>&1 | FileCheck --check-prefix=ABI1 %s
! RUN: %flang -c --target=riscv64-unknown-linux -mabi=lp64f %s -### 2>&1 | FileCheck --check-prefix=ABI2 %s
! RUN: %flang -c --target=riscv64-unknown-linux -mabi=lp64d %s -### 2>&1 | FileCheck --check-prefix=ABI3 %s
! RUN: %flang -c --target=riscv64-unknown-linux %s -### 2>&1 | FileCheck --check-prefix=ABI3 %s

! INVALID1: error: unsupported argument 'ilp32' to option '-mabi='
! INVALID2: error: unsupported argument 'lp64e' to option '-mabi='

! ABI1: "-mabi=lp64"
! ABI2: "-mabi=lp64f"
! ABI3: "-mabi=lp64d"


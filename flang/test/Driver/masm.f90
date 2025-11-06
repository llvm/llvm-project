! RUN: %flang --target=x86_64-unknown-linux -masm=intel -S %s -### 2>&1 | FileCheck --check-prefix=CHECK-INTEL %s
! RUN: %flang --target=x86_64-unknown-linux -masm=att -S %s -### 2>&1 | FileCheck --check-prefix=CHECK-ATT %s
! RUN: not %flang --target=x86_64-unknown-linux -S -masm=somerequired %s -### 2>&1 | FileCheck --check-prefix=CHECK-SOMEREQUIRED %s
! RUN: %flang --target=aarch64-unknown-eabi -S -masm=intel %s -### 2>&1 | FileCheck --check-prefix=CHECK-AARCH64 %s

! CHECK-INTEL: "-mllvm" "-x86-asm-syntax=intel"
! CHECK-ATT: "-mllvm" "-x86-asm-syntax=att"
! CHECK-SOMEREQUIRED: error: unsupported argument 'somerequired' to option '-masm='
! CHECK-AARCH64: warning: argument unused during compilation: '-masm=intel'
! CHECK-AARCH64-NOT: -x86-asm-syntax=intel

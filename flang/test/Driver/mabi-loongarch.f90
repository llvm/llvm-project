! RUN: not %flang -### -c --target=loongarch64-unknown-linux -mabi=lp64s %s 2>&1 | FileCheck --check-prefix=INVALID1 %s
! RUN: not %flang -### -c --target=loongarch64-unknown-linux -mabi=lp64f %s 2>&1 | FileCheck --check-prefix=INVALID2 %s
! RUN: %flang -### -c --target=loongarch64-unknown-linux -mabi=lp64d %s 2>&1 | FileCheck --check-prefix=ABI %s
! RUN: %flang -### -c --target=loongarch64-unknown-linux %s 2>&1 | FileCheck --check-prefix=ABI %s

! REQUIRES: target=loongarch64{{.*}}

! INVALID1: error: invalid argument '-mabi=lp64s'; must support 64-bit fpu for flang in LoongArch64
! INVALID2: error: invalid argument '-mabi=lp64f'; must support 64-bit fpu for flang in LoongArch64

! ABI: "-target-feature" "+d"
! ABI-SAME: "-mabi=lp64d"


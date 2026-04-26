! RUN: not %flang -c --target=loongarch64-unknown-linux -mabi=lp64s %s -### 2>&1 | FileCheck --check-prefix=INVALID1 %s
! RUN: not %flang -c --target=loongarch64-unknown-linux -mabi=lp64f %s -### 2>&1 | FileCheck --check-prefix=INVALID2 %s
! RUN: %flang -c --target=loongarch64-unknown-linux -mabi=lp64d %s -### 2>&1 | FileCheck --check-prefix=ABI %s
! RUN: %flang -c --target=loongarch64-unknown-linux %s -### 2>&1 | FileCheck --check-prefix=ABI %s

! INVALID1: error: invalid argument '-mabi' not allowed with 'lp64s'
! INVALID2: error: invalid argument '-mabi' not allowed with 'lp64f'

! ABI: "-target-feature" "+d"


! Test that flang forwards the -moutline-atomics and -mno-outline-atomics.
! RUN: %flang -moutline-atomics --target=aarch64-none-none -### %s -o %t 2>&1  | FileCheck %s
! CHECK: "-target-feature" "+outline-atomics"

! RUN: %flang -mno-outline-atomics --target=aarch64-none-none -### %s -o %t 2>&1  | FileCheck %s --check-prefix=CHECK-NOOUTLINE
! CHECK-NOOUTLINE: "-target-feature" "-outline-atomics"

! Use Fuchsia to ensure the outline atomics is enabled.
! RUN: %flang --target=aarch64-none-fuchsia -### %s -o %t 2>&1  | FileCheck %s --check-prefix=CHECK-DEFAULT
! CHECK-DEFAULT: "-target-feature" "+outline-atomics"

! RUN: %flang -mno-outline-atomics --target=x86-none-none -### %s -o %t 2>&1  | FileCheck %s --check-prefix=CHECK-ERRMSG
! CHECK-ERRMSG: warning: 'x86' does not support '-mno-outline-atomics'

! Test that flang forwards -moutline-atomics-preserve-most and -mno-outline-atomics-preserve-most.
! RUN: %flang -moutline-atomics-preserve-most --target=aarch64-none-none -### %s -o %t 2>&1 | FileCheck %s --check-prefix=CHECK-PRESERVE-MOST
! CHECK-PRESERVE-MOST: "-target-feature" "+outline-atomics-preserve-most"

! RUN: %flang -mno-outline-atomics-preserve-most --target=aarch64-none-none -### %s -o %t 2>&1 | FileCheck %s --check-prefix=CHECK-NOPRESERVE-MOST
! CHECK-NOPRESERVE-MOST: "-target-feature" "-outline-atomics-preserve-most"

! RUN: %flang -moutline-atomics-preserve-most --target=x86-none-none -### %s -o %t 2>&1 | FileCheck %s --check-prefix=CHECK-PRESERVE-MOST-X86
! CHECK-PRESERVE-MOST-X86: warning: 'x86' does not support '-moutline-atomics-preserve-most'

! RUN: %flang -moutline-atomics-preserve-most --target=arm64ec-pc-windows-msvc -### %s -o %t 2>&1 | FileCheck %s --check-prefix=CHECK-PRESERVE-MOST-ARM64EC
! CHECK-PRESERVE-MOST-ARM64EC: warning: 'arm64ec' does not support '-moutline-atomics-preserve-most'



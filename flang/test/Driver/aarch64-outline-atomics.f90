! Test that flang-new forwards the -moutline-atomics and -mno-outline-atomics.
! RUN: %flang -moutline-atomics --target=aarch64-none-none -### %s -o %t 2>&1  | FileCheck %s
! CHECK: "-target-feature" "+outline-atomics"

! RUN: %flang -mno-outline-atomics --target=aarch64-none-none -### %s -o %t 2>&1  | FileCheck %s --check-prefix=CHECK-NOOUTLINE
! CHECK-NOOUTLINE: "-target-feature" "-outline-atomics"

! Use Fuchsia to ensure the outline atomics is enabled.
! RUN: %flang --target=aarch64-none-fuchsia -### %s -o %t 2>&1  | FileCheck %s --check-prefix=CHECK-DEFAULT
! CHECK-DEFAULT: "-target-feature" "+outline-atomics"

! RUN: %flang -mno-outline-atomics --target=x86-none-none -### %s -o %t 2>&1  | FileCheck %s --check-prefix=CHECK-ERRMSG
! CHECK-ERRMSG: warning: 'x86' does not support '-mno-outline-atomics'



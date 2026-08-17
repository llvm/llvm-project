! Test -f[no-]optimize-sibling-calls driver forwarding to flang -fc1.

! RUN: %flang -### %s 2>&1 | FileCheck %s --check-prefix=CHECK-OPTIMIZE-SIBLING
! RUN: %flang -### -foptimize-sibling-calls %s 2>&1 | FileCheck %s --check-prefix=CHECK-OPTIMIZE-SIBLING-FORWARD
! RUN: %flang -### -fno-optimize-sibling-calls %s 2>&1 | FileCheck %s --check-prefix=CHECK-OPTIMIZE-NOSIBLING
! RUN: %flang -### -fno-optimize-sibling-calls -foptimize-sibling-calls %s 2>&1 | FileCheck %s --check-prefix=CHECK-OPTIMIZE-SIBLING-FORWARD
! RUN: %flang -### -foptimize-sibling-calls -fno-optimize-sibling-calls %s 2>&1 | FileCheck %s --check-prefix=CHECK-OPTIMIZE-NOSIBLING

! CHECK-OPTIMIZE-SIBLING: "-fc1"
! CHECK-OPTIMIZE-SIBLING-NOT: "-fno-optimize-sibling-calls"
! CHECK-OPTIMIZE-SIBLING-NOT: "-foptimize-sibling-calls"

! CHECK-OPTIMIZE-SIBLING-FORWARD: "-fc1"{{.*}}"-foptimize-sibling-calls"
! CHECK-OPTIMIZE-SIBLING-FORWARD-NOT: "-fno-optimize-sibling-calls"

! CHECK-OPTIMIZE-NOSIBLING: "-fc1"{{.*}}"-fno-optimize-sibling-calls"
! CHECK-OPTIMIZE-NOSIBLING-NOT: "-foptimize-sibling-calls"

subroutine test
end subroutine test

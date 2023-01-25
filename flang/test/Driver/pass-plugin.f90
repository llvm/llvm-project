! Verify that the static and dynamically loaded pass plugins work as expected.

! UNSUPPORTED: system-windows

! REQUIRES: plugins, shell, examples

! RUN: %flang -S %s %loadbye -Xflang -fdebug-pass-manager -o /dev/null \
! RUN: 2>&1 | FileCheck %s

! RUN: %flang_fc1 -S %s %loadbye -fdebug-pass-manager -o /dev/null \
! RUN: 2>&1 | FileCheck %s


! CHECK: Running pass: {{.*}}Bye on empty_

subroutine empty
end subroutine empty

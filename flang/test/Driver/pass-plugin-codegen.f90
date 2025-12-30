! Verify that the static and dynamically loaded pass plugins work as expected.

! UNSUPPORTED: system-windows

! REQUIRES: plugins, shell, examples

! RUN: %flang_fc1 -S %s -o - %loadbye \
! RUN: 2>&1 | FileCheck %s --check-prefix=CHECK-INACTIVE

! RUN: %flang_fc1 -S %s -o - %loadbye -mllvm -last-words \
! RUN: | FileCheck %s --check-prefix=CHECK-ACTIVE

! RUN: %flang_fc1 -emit-llvm %s -o - %loadbye -mllvm -last-words \
! RUN: | FileCheck %s --check-prefix=CHECK-LLVM

! RUN: not %flang_fc1 -emit-obj %s -o - %loadbye -mllvm -last-words \
! RUN: 2>&1 | FileCheck %s --check-prefix=CHECK-ERR

! CHECK-INACTIVE-NOT: Bye
! CHECK-INACTIVE: empty_:
! CHECK-ACTIVE: CodeGen Bye
! CHECK-LLVM: define{{.*}} void @empty_
! CHECK-ERR: error: last words unsupported for binary output

subroutine empty
end subroutine empty

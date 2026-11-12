! Verify that frontend and IR output do not require a registered LLVM backend.
! REQUIRES: !sparc-registered-target
! RUN: %flang_fc1 -triple sparc-unknown-unknown -fsyntax-only %s
! RUN: %flang_fc1 -triple sparc-unknown-unknown -emit-hlfir %s -o - | FileCheck %s --check-prefix=FIR
! RUN: %flang_fc1 -triple sparc-unknown-unknown -emit-llvm %s -o - | FileCheck %s
! Exercise the optimized pass middle-end pipeline without a target machine:
! RUN: %flang_fc1 -triple sparc-unknown-unknown -O3 -emit-llvm %s -o - | FileCheck %s
! RUN: %flang_fc1 -triple sparc-unknown-unknown -emit-llvm-bc %s -o /dev/null
! RUN: not %flang_fc1 -triple sparc-unknown-unknown -emit-obj %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERROR
! RUN: not %flang_fc1 -triple sparc-unknown-unknown -S %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERROR

! FIR: llvm.target_triple = "sparc-unknown-unknown"
! CHECK: target triple = "sparc-unknown-unknown"
! CHECK: define {{.*}} @foo_
! ERROR: error: unable to create target:
subroutine foo
end subroutine

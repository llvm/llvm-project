! Verify that frontend and IR output do not require a registered LLVM backend.
! REQUIRES: !x86-registered-target
! RUN: %flang_fc1 -triple x86_64-unknown-linux-gnu -fsyntax-only %s
! RUN: %flang_fc1 -triple x86_64-unknown-linux-gnu -emit-hlfir %s -o - | FileCheck %s --check-prefix=FIR
! RUN: %flang_fc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s
! RUN: %flang_fc1 -triple x86_64-unknown-linux-gnu -O2 -emit-llvm %s -o - | FileCheck %s
! RUN: %flang_fc1 -triple x86_64-unknown-linux-gnu -emit-llvm-bc %s -o %t.bc
! RUN: not %flang_fc1 -triple x86_64-unknown-linux-gnu -emit-obj %s -o %t.o 2>&1 | FileCheck %s --check-prefix=ERROR
! RUN: not %flang_fc1 -triple x86_64-unknown-linux-gnu -S %s -o %t.s 2>&1 | FileCheck %s --check-prefix=ERROR

! FIR: llvm.target_triple = "x86_64-unknown-linux-gnu"
! CHECK: target triple = "x86_64-unknown-linux-gnu"
! CHECK: define {{.*}} @foo_
! ERROR: error: unable to create target:
subroutine foo
end subroutine

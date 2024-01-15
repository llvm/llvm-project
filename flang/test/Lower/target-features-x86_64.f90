! REQUIRES: x86-registered-target
! RUN: %flang_fc1 -emit-fir -triple x86_64-unknown-linux-gnu %s -o - | FileCheck %s --check-prefixes=ALL,NONE
! RUN: %flang_fc1 -emit-fir -triple x86_64-unknown-linux-gnu -target-cpu x86-64 %s -o - | FileCheck %s --check-prefixes=ALL,CPU
! RUN: %flang_fc1 -emit-fir -triple x86_64-unknown-linux-gnu -target-feature +sse %s -o - | FileCheck %s --check-prefixes=ALL,FEATURE
! RUN: %flang_fc1 -emit-fir -triple x86_64-unknown-linux-gnu -target-cpu x86-64 -target-feature +sse %s -o - | FileCheck %s --check-prefixes=ALL,BOTH

! ALL-LABEL: func.func @_QPfoo()

! NONE-NOT: target_cpu
! NONE-NOT: target_features

! CPU-SAME: target_cpu = "x86-64"
! CPU-NOT: target_features

! FEATURE-NOT: target_cpu
! FEATURE-SAME: target_features = #llvm.target_features<["+sse"]>

! BOTH-SAME: target_cpu = "x86-64"
! BOTH-SAME: target_features = #llvm.target_features<["+sse"]>
subroutine foo
end subroutine

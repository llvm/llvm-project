! Test bbc target override.
! REQUIRES: x86-registered-target
! RUN: bbc %s -target x86_64-unknown-linux-gnu -o - | FileCheck %s
subroutine test
end subroutine

! CHECK: module attributes {
! CHECK-SAME: dlti.dl_spec = #dlti.dl_spec<
! CHECK-SAME: llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
! CHECK-SAME: llvm.target_triple = "x86_64-unknown-linux-gnu"

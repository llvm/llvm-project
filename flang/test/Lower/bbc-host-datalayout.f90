! Test bbc set-up of the target data layout from the host.
! RUN: bbc %s -o - | FileCheck %s
subroutine test
end subroutine

! CHECK: module attributes {
! CHECK-SAME: dlti.dl_spec = #dlti.dl_spec<
! CHECK-SAME: llvm.data_layout = "{{[^"]}}
! CHECK-SAME: llvm.target_triple = "{{[^"]}}

! Check that the branch weights used by the array repacking
! are propagated all the way to LLVM IR:
! RUN: %flang_fc1 -frepack-arrays -emit-llvm %s -o - | FileCheck %s

! CHECK-LABEL: define void @test_(
! CHECK-SAME:      ptr noalias [[TMP0:%.*]])
! CHECK:    [[TMP4:%.*]] = ptrtoint ptr [[TMP0]] to i64
! CHECK:    [[TMP5:%.*]] = icmp ne i64 [[TMP4]], 0
! CHECK:    br i1 [[TMP5]], label %[[BB6:.*]], label %[[ABSENT:.*]]
! CHECK:  [[BB6]]:
! CHECK:    [[TMP7:%.*]] = call i1 @_FortranAIsContiguous(ptr [[TMP0]])
! CHECK:    [[TMP8:%.*]] = icmp eq i1 [[TMP7]], false
! CHECK:    [[TMP13:%.*]] = and i1 [[TMP8]], [[TMP12:.*]]
! CHECK:    br i1 [[TMP13]], label %[[BB14:.*]], label %[[NO_REPACK:.*]], !prof [[PROF2:![0-9]+]]
! CHECK:  [[BB14]]:
! CHECK:    call void @_FortranAShallowCopyDirect
! CHECK:    br label %[[BB46:.*]]
! CHECK:  [[NO_REPACK]]:
! CHECK-NEXT: br label %[[BB46]]
! CHECK:  [[BB46]]:
! CHECK:    br label %[[PRESENT_OR_ABSENT:.*]]
! CHECK:  [[ABSENT]]:
! CHECK-NEXT: br label %[[PRESENT_OR_ABSENT]]
! CHECK:  [[PRESENT_OR_ABSENT]]:
! CHECK:    br i1 [[TMP5]], label %[[BB48:.*]], label %[[BB57:.*]]
! CHECK:  [[BB48]]:
! CHECK:    br i1 [[TMP55:.*]], label %[[BB56:.*]], label %[[NO_COPY_BACK:.*]], !prof [[PROF2]]
! CHECK:  [[BB56]]:
! CHECK:    call void @_FortranAShallowCopyDirect
! CHECK:    br label %[[NO_COPY_BACK]]
! CHECK:  [[NO_COPY_BACK]]:
! CHECK-NEXT: br label %[[BB57]]
! CHECK:  [[BB57]]:
! CHECK:    ret void
! CHECK: [[PROF2]] = !{!"branch_weights", i32 0, i32 1}
subroutine test(x)
  real :: x(:)
end subroutine test

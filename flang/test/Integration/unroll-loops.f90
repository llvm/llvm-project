! FIXME: https://github.com/llvm/llvm-project/issues/123668
!
! DEFINE: %{triple} =
! DEFINE: %{check-unroll} = %flang_fc1 -emit-llvm -O1 -funroll-loops -mllvm -force-vector-width=2 -triple %{triple} -o- %s | FileCheck %s --check-prefixes=CHECK,UNROLL
! DEFINE: %{check-nounroll} = %flang_fc1 -emit-llvm -O1 -mllvm -force-vector-width=2 -triple %{triple} -o- %s | FileCheck %s --check-prefixes=CHECK,NO-UNROLL
!
! REDEFINE: %{triple} = aarch64-unknown-linux-gnu
! RUN: %if aarch64-registered-target %{ %{check-unroll} %}
! RUN: %if aarch64-registered-target %{ %{check-nounroll} %}
!
! REDEFINE: %{triple} = x86_64-unknown-linux-gnu
! RUN: %if x86-registered-target %{ %{check-unroll} %}
! RUN: %if x86-registered-target %{ %{check-nounroll} %}
!
! CHECK-LABEL: @unroll
! CHECK-SAME: (ptr writeonly captures(none) %[[ARG0:.*]])
subroutine unroll(a)
  integer(kind=8), intent(out) :: a(1000)
  integer(kind=8) :: i
    ! CHECK: br label %[[BLK:.*]]
    ! CHECK: [[BLK]]:
    ! CHECK-NEXT: %[[IND:.*]] = phi i64 [ 0, %{{.*}} ], [ %[[NIV:.*]], %[[BLK]] ]
    ! CHECK-NEXT: %[[VIND:.*]] = phi <2 x i64> [ <i64 1, i64 2>, %{{.*}} ], [ %[[NVIND:.*]], %[[BLK]] ]
    !
    ! NO-UNROLL-NEXT: %[[GEP:.*]] = getelementptr i64, ptr %[[ARG0]], i64 %[[IND]]
    ! NO-UNROLL-NEXT: store <2 x i64> %[[VIND]], ptr %[[GEP]]
    ! NO-UNROLL-NEXT: %[[NIV:.*]] = add nuw i64 %{{.*}}, 2
    ! NO-UNROLL-NEXT: %[[NVIND]] = add <2 x i64> %[[VIND]], splat (i64 2)
    !
    ! UNROLL-NEXT: %[[VIND1:.*]] = add <2 x i64> %[[VIND]], splat (i64 2)
    ! UNROLL-NEXT: %[[GEP0:.*]] = getelementptr i64, ptr %[[ARG0]], i64 %[[IND]]
    ! UNROLL-NEXT: %[[GEP1:.*]] = getelementptr i8, ptr %[[GEP0]], i64 16
    ! UNROLL-NEXT: store <2 x i64> %[[VIND]], ptr %[[GEP0]]
    ! UNROLL-NEXT: store <2 x i64> %[[VIND1]], ptr %[[GEP1]]
    ! UNROLL-NEXT: %[[NIV:.*]] = add nuw i64 %[[IND]], 4
    ! UNROLL-NEXT: %[[NVIND:.*]] = add <2 x i64> %[[VIND]], splat (i64 4)
    !
    ! CHECK-NEXT: %[[EXIT:.*]] = icmp eq i64 %[[NIV]], 1000
    ! CHECK-NEXT: br i1 %[[EXIT]], label %{{.*}}, label %[[BLK]]
  do i=1,1000
    a(i) = i
  end do
end subroutine

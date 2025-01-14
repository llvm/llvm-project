! RUN: %flang_fc1 -emit-llvm -O1 -funroll-loops -mllvm -force-vector-width=2 -o- %s | FileCheck %s --check-prefixes=CHECK,UNROLL
! RUN: %flang_fc1 -emit-llvm -O2 -mllvm -force-vector-width=2 -o- %s | FileCheck %s --check-prefixes=CHECK,UNROLL
! RUN: %flang_fc1 -emit-llvm -O1 -fno-unroll-loops -mllvm -force-vector-width=2 -o- %s | FileCheck %s --check-prefixes=CHECK,NO-UNROLL
! RUN: %flang_fc1 -emit-llvm -O1 -mllvm -force-vector-width=2 -o- %s | FileCheck %s --check-prefixes=CHECK,NO-UNROLL

! CHECK-LABEL: @unroll
! CHECK-SAME: (ptr nocapture writeonly %[[ARG0:.*]])
subroutine unroll(a)
  real(kind=8), intent(out) :: a(1000)
  integer(kind=8) :: i
    ! CHECK: br label %[[BLK:.*]]
    ! CHECK: [[BLK]]:
    ! CHECK-NEXT: %[[IND:.*]] = phi i64 [ 0, %{{.*}} ], [ %[[NIV:.*]], %[[BLK]] ]
    ! CHECK-NEXT: %[[VIND:.*]] = phi <2 x i64> [ <i64 1, i64 2>, %{{.*}} ], [ %[[NVIND:.*]], %[[BLK]] ]
    !
    ! NO-UNROLL-NEXT: %[[IV_D:.*]] = uitofp nneg <2 x i64> %[[VIND]] to <2 x double>
    ! NO-UNROLL-NEXT: %[[GEP:.*]] = getelementptr double, ptr %[[ARG0]], i64 %[[IND]]
    ! NO-UNROLL-NEXT: store <2 x double> %[[IV_D]], ptr %[[GEP]]
    ! NO-UNROLL-NEXT: %[[NIV:.*]] = add nuw i64 %{{.*}}, 2
    ! NO-UNROLL-NEXT: %[[NVIND]] = add <2 x i64> %[[VIND]], splat (i64 2)
    !
    ! UNROLL-NEXT: %[[VIND1:.*]] = add <2 x i64> %[[VIND]], splat (i64 2)
    ! UNROLL-NEXT: %[[IV0_D:.*]] = uitofp nneg <2 x i64> %[[VIND]] to <2 x double>
    ! UNROLL-NEXT: %[[IV1_D:.*]] = uitofp nneg <2 x i64> %[[VIND1]] to <2 x double>
    ! UNROLL-NEXT: %[[GEP0:.*]] = getelementptr double, ptr %[[ARG0]], i64 %[[IND]]
    ! UNROLL-NEXT: %[[GEP1:.*]] = getelementptr i8, ptr %[[GEP0]], i64 16
    ! UNROLL-NEXT: store <2 x double> %[[IV0_D]], ptr %[[GEP0]]
    ! UNROLL-NEXT: store <2 x double> %[[IV1_D]], ptr %[[GEP1]]
    ! UNROLL-NEXT: %[[NIV:.*]] = add nuw i64 %[[IND]], 4
    ! UNROLL-NEXT: %[[NVIND:.*]] = add <2 x i64> %[[VIND]], splat (i64 4)
    !
    ! CHECK-NEXT: %[[EXIT:.*]] = icmp eq i64 %[[NIV]], 1000
    ! CHECK-NEXT: br i1 %[[EXIT]], label %{{.*}}, label %[[BLK]]
  do i=1,1000
    a(i) = i
  end do
end subroutine

! Tests that -finit-local= does not initialize Cray pointees. A Cray pointee
! has no storage of its own; its FIR base is a pointer-box descriptor. Before
! this fix, shouldInitLocal admitted the pointee and the scalar-fallback path
! in initAddr emitted a memcpy from null into the descriptor. With -O2 this
! caused the function to be optimized to `unreachable`.
!
! RUN: %flang_fc1 -emit-llvm -O0 -finit-local=zero %s -o - | FileCheck --check-prefix=O0 %s
! RUN: %flang_fc1 -emit-llvm -O2 -finit-local=zero %s -o - | FileCheck --check-prefix=O2 %s

! The pointee x must NOT be initialized; the only store must be the user
! assignment x(3) = 7 (i32 7).  No memcpy from null and no zeroinitializer.

subroutine test_cray_pointee(res)
  integer :: res(10), x(10)
  integer(8) :: p
  pointer (p, x)
  p = loc(res)
  x(3) = 7
  res = x
end subroutine

! O0-LABEL: define {{.*}}@{{.*}}test_cray_pointee{{.*}}(
! O0-NOT:  call void @llvm.memcpy{{.*}}null
! O0-NOT:  store {{.*}} zeroinitializer
! O0:      store i32 7,

! O2-LABEL: define {{.*}}@{{.*}}test_cray_pointee{{.*}}(
! O2-NOT:  unreachable
! O2-NOT:  call void @llvm.memcpy{{.*}}null
! O2:      store i32 7,

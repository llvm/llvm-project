! RUN: %flang_fc1 -emit-hlfir -fopenmp -o - %s | FileCheck %s
! RUN: bbc -emit-hlfir -fopenmp -o - %s | FileCheck %s

! Allocatable privatizers need initialization, copying and cleanup. References
! after the single construct must resolve to the original variables again.
! CHECK-LABEL: func.func @_QPsingle_allocatables(
! CHECK: %[[A:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFsingle_allocatablesEa")
! CHECK: %[[B:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFsingle_allocatablesEb")
! CHECK: omp.single private(@{{[^ ]+}} %[[A]]#0 -> %[[PA:[^,]+]], @{{[^ ]+}} %[[B]]#0 -> %[[PB:[^ ]+]] :
! CHECK: %[[DA:.*]]:2 = hlfir.declare %[[PA]] uniq_name("_QFsingle_allocatablesEa")
! CHECK: %[[DB:.*]]:2 = hlfir.declare %[[PB]] uniq_name("_QFsingle_allocatablesEb")
! CHECK: fir.call @_QPuse_allocatables(%[[DA]]#0, %[[DB]]#0)
! CHECK: omp.terminator
! CHECK-NEXT: }
! CHECK: fir.call @_QPuse_allocatables(%[[A]]#0, %[[B]]#0)
subroutine single_allocatables(a, b)
  integer, allocatable :: a(:), b(:)
  interface
    subroutine use_allocatables(a, b)
      integer, allocatable :: a(:), b(:)
    end subroutine
  end interface
  !$omp single private(a) firstprivate(b)
    call use_allocatables(a, b)
  !$omp end single
  call use_allocatables(a, b)
end subroutine

! CHECK-LABEL: func.func @_QPsingle_nowait_private(
! CHECK: %[[X:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFsingle_nowait_privateEx")
! CHECK: %[[Y:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFsingle_nowait_privateEy")
! CHECK: omp.single nowait private(@{{[^ ]+}} %[[X]]#0 -> %[[PX:[^,]+]], @{{[^ ]+}} %[[Y]]#0 -> %[[PY:[^ ]+]] : !fir.ref<i32>, !fir.ref<i32>) {
! CHECK-NEXT: %[[DX:.*]]:2 = hlfir.declare %[[PX]]
! CHECK-NEXT: %[[DY:.*]]:2 = hlfir.declare %[[PY]]
! CHECK: %[[VALUE:.*]] = fir.load %[[DY]]#0
! CHECK: hlfir.assign %[[VALUE]] to %[[DX]]#0
! CHECK: omp.terminator
! CHECK-NEXT: }
! CHECK: %[[ORIGINAL:.*]] = fir.load %[[Y]]#0
! CHECK: hlfir.assign %[[ORIGINAL]] to %[[X]]#0
subroutine single_nowait_private(x, y)
  integer :: x, y
  !$omp single private(x) firstprivate(y)
    x = y
  !$omp end single nowait
  x = y
end subroutine

! CHECK-LABEL: func.func @_QPsingle_copyprivate_private(
! CHECK: omp.parallel private(@{{[^ ]+}} %{{.*}} -> %[[Z:[^ ]+]] : !fir.ref<i32>) {
! CHECK: %[[DZ:.*]]:2 = hlfir.declare %[[Z]]
! CHECK: omp.single copyprivate(%[[DZ]]#0 -> @_copy_ref_i32 : !fir.ref<i32>) private(
! CHECK-SAME: @_QFsingle_copyprivate_privateEx_private_i32 %{{.*}} -> %[[PX:[^,]+]], @_QFsingle_copyprivate_privateEy_firstprivate_i32 %{{.*}} -> %[[PY:[^ ]+]] : !fir.ref<i32>, !fir.ref<i32>) {
! CHECK-NEXT: %[[DX:.*]]:2 = hlfir.declare %[[PX]]
! CHECK-NEXT: %[[DY:.*]]:2 = hlfir.declare %[[PY]]
! CHECK: fir.call @_QPuse_scalars(%[[DX]]#0, %[[DY]]#0, %[[DZ]]#0)
subroutine single_copyprivate_private(x, y, z)
  integer :: x, y, z
  !$omp parallel private(z)
    !$omp single private(x) firstprivate(y)
      call use_scalars(x, y, z)
    !$omp end single copyprivate(z)
  !$omp end parallel
end subroutine

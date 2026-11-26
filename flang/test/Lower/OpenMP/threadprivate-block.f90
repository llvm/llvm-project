! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s
! RUN: bbc -emit-hlfir -fopenmp %s -o - | FileCheck %s

! Regression tests for https://github.com/llvm/llvm-project/issues/230312.
! Threadprivate variables declared in nested BLOCKs are not instantiated at
! parallel-region entry. Their threadprivate operations must be generated when
! lowering the BLOCKs, and assignments must use the resulting thread-local
! addresses. Use stacksave/stackrestore to check the BLOCK boundaries.

! CHECK-LABEL: func.func @_QPblock_in_parallel()
! CHECK-NOT: omp.threadprivate
! CHECK: omp.parallel {
! CHECK-NOT: omp.threadprivate
! CHECK: %[[BLOCK_SP:.*]] = llvm.intr.stacksave : !llvm.ptr
! CHECK: %[[ADDR:.*]] = fir.address_of(@{{.*}}) : !fir.ref<i32>
! CHECK: %[[DECL:.*]]:2 = hlfir.declare %[[ADDR]]
! CHECK: %[[TP:.*]] = omp.threadprivate %[[DECL]]#0 : !fir.ref<i32> -> !fir.ref<i32>
! CHECK: %[[TP_DECL:.*]]:2 = hlfir.declare %[[TP]]
! CHECK: hlfir.assign %{{.*}} to %[[TP_DECL]]#0 : i32, !fir.ref<i32>
! CHECK: llvm.intr.stackrestore %[[BLOCK_SP]] : !llvm.ptr
! CHECK: omp.terminator
subroutine block_in_parallel()
  implicit none
  !$omp parallel
  block
    integer, save :: tp
    !$omp threadprivate(tp)
    tp = 1
  end block
  !$omp end parallel
end subroutine

! CHECK-LABEL: func.func @_QPnested_blocks_in_parallel()
! CHECK-NOT: omp.threadprivate
! CHECK: omp.parallel {
! CHECK-NOT: omp.threadprivate
! CHECK: %[[OUTER_SP:.*]] = llvm.intr.stacksave : !llvm.ptr
! CHECK-NOT: omp.threadprivate
! CHECK: %[[INNER_SP:.*]] = llvm.intr.stacksave : !llvm.ptr
! CHECK: %[[ADDR:.*]] = fir.address_of(@{{.*}}) : !fir.ref<i32>
! CHECK: %[[DECL:.*]]:2 = hlfir.declare %[[ADDR]]
! CHECK: %[[TP:.*]] = omp.threadprivate %[[DECL]]#0 : !fir.ref<i32> -> !fir.ref<i32>
! CHECK: %[[TP_DECL:.*]]:2 = hlfir.declare %[[TP]]
! CHECK: hlfir.assign %{{.*}} to %[[TP_DECL]]#0 : i32, !fir.ref<i32>
! CHECK: llvm.intr.stackrestore %[[INNER_SP]] : !llvm.ptr
! CHECK: llvm.intr.stackrestore %[[OUTER_SP]] : !llvm.ptr
! CHECK: omp.terminator
subroutine nested_blocks_in_parallel()
  implicit none
  !$omp parallel
  block
    block
      integer, save :: tp
      !$omp threadprivate(tp)
      tp = 1
    end block
  end block
  !$omp end parallel
end subroutine

! CHECK-LABEL: func.func @_QPblock_in_parallel_do()
! CHECK-NOT: omp.threadprivate
! CHECK: omp.parallel {
! CHECK-NOT: omp.threadprivate
! CHECK: omp.wsloop
! CHECK: omp.loop_nest
! CHECK-NOT: omp.threadprivate
! CHECK: %[[BLOCK_SP:.*]] = llvm.intr.stacksave : !llvm.ptr
! CHECK: %[[ADDR:.*]] = fir.address_of(@{{.*}}) : !fir.ref<i32>
! CHECK: %[[DECL:.*]]:2 = hlfir.declare %[[ADDR]]
! CHECK: %[[TP:.*]] = omp.threadprivate %[[DECL]]#0 : !fir.ref<i32> -> !fir.ref<i32>
! CHECK: %[[TP_DECL:.*]]:2 = hlfir.declare %[[TP]]
! CHECK: hlfir.assign %{{.*}} to %[[TP_DECL]]#0 : i32, !fir.ref<i32>
! CHECK: llvm.intr.stackrestore %[[BLOCK_SP]] : !llvm.ptr
! CHECK: omp.yield
! CHECK: omp.terminator
subroutine block_in_parallel_do()
  implicit none
  integer :: i
  !$omp parallel do
  do i = 1, 10
    block
      integer, save :: tp
      !$omp threadprivate(tp)
      tp = i
    end block
  end do
  !$omp end parallel do
end subroutine

! CHECK-LABEL: func.func @_QPblock_in_task()
! CHECK-NOT: omp.threadprivate
! CHECK: omp.parallel {
! CHECK: omp.single
! CHECK: omp.task
! CHECK-NOT: omp.threadprivate
! CHECK: %[[BLOCK_SP:.*]] = llvm.intr.stacksave : !llvm.ptr
! CHECK: %[[ADDR:.*]] = fir.address_of(@{{.*}}) : !fir.ref<i32>
! CHECK: %[[DECL:.*]]:2 = hlfir.declare %[[ADDR]]
! CHECK: %[[TP:.*]] = omp.threadprivate %[[DECL]]#0 : !fir.ref<i32> -> !fir.ref<i32>
! CHECK: %[[TP_DECL:.*]]:2 = hlfir.declare %[[TP]]
! CHECK: hlfir.assign %{{.*}} to %[[TP_DECL]]#0 : i32, !fir.ref<i32>
! CHECK: llvm.intr.stackrestore %[[BLOCK_SP]] : !llvm.ptr
! CHECK: omp.terminator
subroutine block_in_task()
  implicit none
  !$omp parallel
  !$omp single
  !$omp task
  block
    integer, save :: tp
    !$omp threadprivate(tp)
    tp = 1
  end block
  !$omp end task
  !$omp end single
  !$omp end parallel
end subroutine

! The outer parallel must defer handling tp until the BLOCK is instantiated.
! The inner parallel must then rebind the already instantiated symbol.
! CHECK-LABEL: func.func @_QPblock_with_inner_parallel()
! CHECK-NOT: omp.threadprivate
! CHECK: omp.parallel {
! CHECK-NOT: omp.threadprivate
! CHECK: %[[BLOCK_SP:.*]] = llvm.intr.stacksave : !llvm.ptr
! CHECK: %[[ADDR:.*]] = fir.address_of(@{{.*}}) : !fir.ref<i32>
! CHECK: %[[DECL:.*]]:2 = hlfir.declare %[[ADDR]]
! CHECK: %[[TP:.*]] = omp.threadprivate %[[DECL]]#0 : !fir.ref<i32> -> !fir.ref<i32>
! CHECK: %[[TP_DECL:.*]]:2 = hlfir.declare %[[TP]]
! CHECK: omp.parallel {
! CHECK: %[[INNER_TP:.*]] = omp.threadprivate %[[DECL]]#0 : !fir.ref<i32> -> !fir.ref<i32>
! CHECK: %[[INNER_TP_DECL:.*]]:2 = hlfir.declare %[[INNER_TP]]
! CHECK: hlfir.assign %{{.*}} to %[[INNER_TP_DECL]]#0 : i32, !fir.ref<i32>
! CHECK: omp.terminator
! CHECK: }
! CHECK: llvm.intr.stackrestore %[[BLOCK_SP]] : !llvm.ptr
! CHECK: omp.terminator
subroutine block_with_inner_parallel()
  implicit none
  !$omp parallel
  block
    integer, save :: tp
    !$omp threadprivate(tp)
    !$omp parallel
      tp = 1
    !$omp end parallel
  end block
  !$omp end parallel
end subroutine

! Declaration initialization gives tp an implicit SAVE attribute.
! CHECK-LABEL: func.func @_QPblock_with_implied_save()
! CHECK-NOT: omp.threadprivate
! CHECK: omp.parallel {
! CHECK-NOT: omp.threadprivate
! CHECK: %[[BLOCK_SP:.*]] = llvm.intr.stacksave : !llvm.ptr
! CHECK: %[[ADDR:.*]] = fir.address_of(@{{.*}}) : !fir.ref<i32>
! CHECK: %[[DECL:.*]]:2 = hlfir.declare %[[ADDR]]
! CHECK: %[[TP:.*]] = omp.threadprivate %[[DECL]]#0 : !fir.ref<i32> -> !fir.ref<i32>
! CHECK: %[[TP_DECL:.*]]:2 = hlfir.declare %[[TP]]
! CHECK: fir.load %[[TP_DECL]]#0 : !fir.ref<i32>
! CHECK: hlfir.assign %{{.*}} to %[[TP_DECL]]#0 : i32, !fir.ref<i32>
! CHECK: llvm.intr.stackrestore %[[BLOCK_SP]] : !llvm.ptr
! CHECK: omp.terminator
subroutine block_with_implied_save()
  implicit none
  !$omp parallel
  block
    integer :: tp = 0
    !$omp threadprivate(tp)
    tp = tp + 1
  end block
  !$omp end parallel
end subroutine

! An enclosing BLOCK has already instantiated tp at parallel-region entry.
! Check that it is rebound inside the parallel and that the enclosing binding
! is restored afterwards.
! CHECK-LABEL: func.func @_QPblock_enclosing_parallel()
! CHECK: %[[BLOCK_SP:.*]] = llvm.intr.stacksave : !llvm.ptr
! CHECK: %[[ADDR:.*]] = fir.address_of(@{{.*}}) : !fir.ref<i32>
! CHECK: %[[DECL:.*]]:2 = hlfir.declare %[[ADDR]]
! CHECK: %[[TP:.*]] = omp.threadprivate %[[DECL]]#0 : !fir.ref<i32> -> !fir.ref<i32>
! CHECK: %[[TP_DECL:.*]]:2 = hlfir.declare %[[TP]]
! CHECK: hlfir.assign %{{.*}} to %[[TP_DECL]]#0 : i32, !fir.ref<i32>
! CHECK: omp.parallel {
! CHECK: %[[INNER_TP:.*]] = omp.threadprivate %[[DECL]]#0 : !fir.ref<i32> -> !fir.ref<i32>
! CHECK: %[[INNER_TP_DECL:.*]]:2 = hlfir.declare %[[INNER_TP]]
! CHECK: hlfir.assign %{{.*}} to %[[INNER_TP_DECL]]#0 : i32, !fir.ref<i32>
! CHECK: omp.terminator
! CHECK: }
! CHECK: hlfir.assign %{{.*}} to %[[TP_DECL]]#0 : i32, !fir.ref<i32>
! CHECK: llvm.intr.stackrestore %[[BLOCK_SP]] : !llvm.ptr
! CHECK: return
subroutine block_enclosing_parallel()
  implicit none
  block
    integer, save :: tp
    !$omp threadprivate(tp)
    tp = 1
    !$omp parallel
      tp = 2
    !$omp end parallel
    tp = 3
  end block
end subroutine

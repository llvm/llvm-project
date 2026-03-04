!===----------------------------------------------------------------------===!
! This directory can be used to add Integration tests involving multiple
! stages of the compiler (for eg. from Fortran to LLVM IR). It should not
! contain executable tests. We should only add tests here sparingly and only
! if there is no other way to test. Repeat this message in each test that is
! added to this directory and sub-directories.
!===----------------------------------------------------------------------===!

!RUN: %flang_fc1 -triple x86_64-unknown-linux-gnu -emit-llvm -fopenmp -fopenmp-version=51 %s -o - | FileCheck %s

! Int "==" → cmpxchg, default (monotonic) ordering
!CHECK-LABEL: define void @atomic_compare_integer_(
!CHECK-SAME: ptr noalias %[[X:.*]], ptr noalias %[[E:.*]], ptr noalias %[[D:.*]])
!CHECK: %[[EVAL:.*]] = load i32, ptr %[[E]], align 4
!CHECK: %[[DVAL:.*]] = load i32, ptr %[[D]], align 4
!CHECK: cmpxchg ptr %[[X]], i32 %[[EVAL]], i32 %[[DVAL]] monotonic monotonic
subroutine atomic_compare_integer(x, e, d)
  integer :: x, e, d
  !$omp atomic compare
  if (x == e) x = d
end

! seq_cst ordering → cmpxchg seq_cst + flush
!CHECK-LABEL: define void @atomic_compare_seq_cst_(
!CHECK-SAME: ptr noalias %[[X:.*]], ptr noalias %[[E:.*]], ptr noalias %[[D:.*]])
!CHECK: %[[EVAL:.*]] = load i32, ptr %[[E]], align 4
!CHECK: %[[DVAL:.*]] = load i32, ptr %[[D]], align 4
!CHECK: cmpxchg ptr %[[X]], i32 %[[EVAL]], i32 %[[DVAL]] seq_cst seq_cst
!CHECK: call void @__kmpc_flush(
subroutine atomic_compare_seq_cst(x, e, d)
  integer :: x, e, d
  !$omp atomic compare seq_cst
  if (x == e) x = d
end

! acquire ordering → cmpxchg acquire
!CHECK-LABEL: define void @atomic_compare_acquire_(
!CHECK-SAME: ptr noalias %[[X:.*]], ptr noalias %[[E:.*]], ptr noalias %[[D:.*]])
!CHECK: %[[EVAL:.*]] = load i32, ptr %[[E]], align 4
!CHECK: %[[DVAL:.*]] = load i32, ptr %[[D]], align 4
!CHECK: cmpxchg ptr %[[X]], i32 %[[EVAL]], i32 %[[DVAL]] acquire acquire
subroutine atomic_compare_acquire(x, e, d)
  integer :: x, e, d
  !$omp atomic compare acquire
  if (x == e) x = d
end

! release ordering → cmpxchg release + flush
!CHECK-LABEL: define void @atomic_compare_release_(
!CHECK-SAME: ptr noalias %[[X:.*]], ptr noalias %[[E:.*]], ptr noalias %[[D:.*]])
!CHECK: %[[EVAL:.*]] = load i32, ptr %[[E]], align 4
!CHECK: %[[DVAL:.*]] = load i32, ptr %[[D]], align 4
!CHECK: cmpxchg ptr %[[X]], i32 %[[EVAL]], i32 %[[DVAL]] release monotonic
!CHECK: call void @__kmpc_flush(
subroutine atomic_compare_release(x, e, d)
  integer :: x, e, d
  !$omp atomic compare release
  if (x == e) x = d
end

! relaxed ordering → cmpxchg monotonic
!CHECK-LABEL: define void @atomic_compare_relaxed_(
!CHECK-SAME: ptr noalias %[[X:.*]], ptr noalias %[[E:.*]], ptr noalias %[[D:.*]])
!CHECK: %[[EVAL:.*]] = load i32, ptr %[[E]], align 4
!CHECK: %[[DVAL:.*]] = load i32, ptr %[[D]], align 4
!CHECK: cmpxchg ptr %[[X]], i32 %[[EVAL]], i32 %[[DVAL]] monotonic monotonic
subroutine atomic_compare_relaxed(x, e, d)
  integer :: x, e, d
  !$omp atomic compare relaxed
  if (x == e) x = d
end

! Less-than comparison → atomicrmw umax
!CHECK-LABEL: define void @atomic_compare_lt_(
!CHECK-SAME: ptr noalias %[[X:.*]], ptr noalias %[[E:.*]])
!CHECK: %[[EVAL:.*]] = load i32, ptr %[[E]], align 4
!CHECK: atomicrmw umax ptr %[[X]], i32 %[[EVAL]] monotonic
subroutine atomic_compare_lt(x, e)
  integer :: x, e
  !$omp atomic compare
  if (x < e) x = e
end

! Less-than with seq_cst → atomicrmw umax seq_cst + flush
!CHECK-LABEL: define void @atomic_compare_lt_seq_cst_(
!CHECK-SAME: ptr noalias %[[X:.*]], ptr noalias %[[E:.*]])
!CHECK: %[[EVAL:.*]] = load i32, ptr %[[E]], align 4
!CHECK: atomicrmw umax ptr %[[X]], i32 %[[EVAL]] seq_cst
!CHECK: call void @__kmpc_flush(
subroutine atomic_compare_lt_seq_cst(x, e)
  integer :: x, e
  !$omp atomic compare seq_cst
  if (x < e) x = e
end

! Less-than with acquire → atomicrmw umax acquire
!CHECK-LABEL: define void @atomic_compare_lt_acquire_(
!CHECK-SAME: ptr noalias %[[X:.*]], ptr noalias %[[E:.*]])
!CHECK: %[[EVAL:.*]] = load i32, ptr %[[E]], align 4
!CHECK: atomicrmw umax ptr %[[X]], i32 %[[EVAL]] acquire
subroutine atomic_compare_lt_acquire(x, e)
  integer :: x, e
  !$omp atomic compare acquire
  if (x < e) x = e
end

! Greater-than comparison → atomicrmw umin
!CHECK-LABEL: define void @atomic_compare_gt_(
!CHECK-SAME: ptr noalias %[[X:.*]], ptr noalias %[[E:.*]])
!CHECK: %[[EVAL:.*]] = load i32, ptr %[[E]], align 4
!CHECK: atomicrmw umin ptr %[[X]], i32 %[[EVAL]] monotonic
subroutine atomic_compare_gt(x, e)
  integer :: x, e
  !$omp atomic compare
  if (x > e) x = e
end

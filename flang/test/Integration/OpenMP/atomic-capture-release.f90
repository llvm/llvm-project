!===----------------------------------------------------------------------===!
! This directory can be used to add Integration tests involving multiple
! stages of the compiler (for eg. from Fortran to LLVM IR). It should not
! contain executable tests. We should only add tests here sparingly and only
! if there is no other way to test. Repeat this message in each test that is
! added to this directory and sub-directories.
!===----------------------------------------------------------------------===!

!RUN: %flang_fc1 -triple x86_64-unknown-linux-gnu -emit-llvm -fopenmp -fopenmp-version=51 %s -o - | FileCheck %s

! Real(4) capture with release ordering: the initial load in the cmpxchg loop
! must use monotonic (not release, which is invalid for loads in LLVM IR).
!CHECK: define void {{.*}}test_capture_release_(
!CHECK-SAME: ptr noalias %[[A:.*]], ptr noalias %[[B:.*]], ptr noalias %[[C:.*]])
!CHECK: [[ENTRY:.*]]:
!CHECK: [[ATOMIC_LOAD:.*]] = load atomic i32, ptr %[[A]] monotonic, align 4
!CHECK: [[CONT:.*]]:
!CHECK: cmpxchg ptr %[[A]], i32 %{{.*}}, i32 %{{.*}} release monotonic
!CHECK: [[EXIT:.*]]:
!CHECK: call void {{.*}}__kmpc_flush{{.*}}
!CHECK: ret
subroutine test_capture_release(a,b,c)
  real(4) :: a, b, c
  !$omp atomic capture release
  c = a
  a = a + b
  !$omp end atomic
end subroutine

! Integer(4) capture with acq_rel ordering: uses atomicrmw (valid with acq_rel).
!CHECK: define void {{.*}}test_capture_acq_rel_(
!CHECK-SAME: ptr noalias %[[A:.*]], ptr noalias %[[B:.*]], ptr noalias %[[C:.*]])
!CHECK: %[[TMP1:.*]] = load {{.*}}, ptr %[[B]]
!CHECK: [[ENTRY:.*]]:
!CHECK: %[[TMP2:.*]] = atomicrmw add ptr %[[A]], i32 %[[TMP1]] acq_rel
!CHECK: call void {{.*}}__kmpc_flush{{.*}}
!CHECK: ret
subroutine test_capture_acq_rel(a,b,c)
  integer(4) :: a, b, c
  !$omp atomic capture acq_rel
  c = a
  a = a + b
  !$omp end atomic
end subroutine

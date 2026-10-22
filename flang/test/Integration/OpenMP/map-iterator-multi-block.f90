!===----------------------------------------------------------------------===!
! This directory can be used to add Integration tests involving multiple
! stages of the compiler (for eg. from Fortran to LLVM IR). It should not
! contain executable tests. We should only add tests here sparingly and only
! if there is no other way to test. Repeat this message in each test that is
! added to this directory and sub-directories.
!===----------------------------------------------------------------------===!

! Lowering a scalar subscript that itself involves control flow (here,
! LEN_TRIM) produces an `omp.iterator` region with several blocks instead of
! one ending in `omp.yield`. An HLFIR-only test would not exercise this, since
! the control flow is introduced by a later lowering pass. This exercises the
! full pipeline down to LLVM IR translation of the generated user-defined
! mapper.

! RUN: %flang_fc1 -emit-llvm -fopenmp -fopenmp-version=52 -fopenmp-targets=amdgcn-amd-amdhsa -o - %s | FileCheck %s

subroutine f(x)
  type :: t
    integer :: a(20)
    character(10) :: text
  end type
  type(t) :: x

  !$omp declare mapper(m: t :: v) &
  !$omp& map(iterator(i=1:2): v%a(i+len_trim(v%text)))
  !$omp target data map(mapper(m): x)
  !$omp end target data
end

! CHECK-LABEL: define internal void @.omp_mapper._QQFfm
! CHECK: omp_mapper.iterator.body:
! CHECK: br label %omp.iterator.region
! The LEN_TRIM scan is a loop inside the iterator region, with its own
! separate blocks ahead of the one that computes the mapped address.
! CHECK: omp.iterator.region1:
! CHECK: omp.iterator.region2:
! CHECK: omp.iterator.region3:
! CHECK: %[[ADDR:.*]] = getelementptr {{.*}} i32, ptr %{{.*}}, i64 %{{.*}}
! CHECK: br label %omp.iterator.region.cont
! CHECK: omp.iterator.region.cont:
! CHECK: call void @__tgt_push_mapper_component(ptr %{{[0-9]+}}, ptr %[[ADDR]], ptr %[[ADDR]], i64 4, i64 %{{.*}}, ptr null)

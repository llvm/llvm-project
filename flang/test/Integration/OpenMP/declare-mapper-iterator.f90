!===----------------------------------------------------------------------===!
! This directory can be used to add Integration tests involving multiple
! stages of the compiler (for eg. from Fortran to LLVM IR). It should not
! contain executable tests. We should only add tests here sparingly and only
! if there is no other way to test. Repeat this message in each test that is
! added to this directory and sub-directories.
!===----------------------------------------------------------------------===!

! A declare mapper's map clause with an iterator modifier lowers (via the
! shared map/motion iterator lowering) to a per-iteration, bounds-described
! omp.map.info. This exercises the user-defined-mapper LLVM IR generation for
! that shape end to end: the array base from var_ptr, indexed by the bounds'
! lower_bound, and pushed via __tgt_push_mapper_component since the iterator's
! trip count is only known at runtime.

! RUN: %flang_fc1 -emit-llvm -fopenmp -fopenmp-version=52 -fopenmp-targets=amdgcn-amd-amdhsa -o - %s | FileCheck %s

subroutine f(arg)
  type :: s
    integer :: a(10)
  end type
  type(s) :: arg

  !$omp declare mapper(m: s :: v) map(mapper(m), iterator(i = 1:10): v%a(i))
  !$omp target data map(mapper(m): arg)
  !$omp end target data
end

! CHECK-LABEL: define internal void @.omp_mapper._QQFfm
! CHECK: omp_mapper.iterator.cond:
! CHECK: icmp ult i64 %{{.*}}, 10
! CHECK: omp_mapper.iterator.body:
! CHECK: %[[FIELD:.*]] = getelementptr %_QFfTs, ptr %omp.arraymap.ptrcurrent, i32 0, i32 0
! CHECK: %[[ELEM:.*]] = getelementptr inbounds [10 x i32], ptr %[[FIELD]], i64 0, i64 %{{.*}}
! CHECK: call void @__tgt_push_mapper_component(ptr %{{.*}}, ptr %[[ELEM]], ptr %[[ELEM]], i64 4, i64 %{{.*}}, ptr null)

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
    integer :: n
    integer :: a(10)
  end type
  type(s) :: arg

  !$omp declare mapper(m: s :: v) map(iterator(i = 1:v%n): v%a(i))
  !$omp target data map(mapper(m): arg)
  !$omp end target data
end

! CHECK-LABEL: define internal void @.omp_mapper._QQFfm
! The iterator variable's storage and the span accumulators are hoisted out
! of the per-element loop.
! CHECK-NEXT: entry:
! CHECK-NEXT: alloca i32
! CHECK-NEXT: %omp.iterparent.begin = alloca i64
! CHECK-NEXT: %omp.iterparent.end = alloca i64
! CHECK-NOT: alloca
! A pre-pass over the iterations finds the span of the elements; a
! storage-only parent for exactly that span is pushed only when the runtime
! trip count is non-zero, then each element is pushed MEMBER_OF it.
! CHECK: %[[HAS_TRIPS:.*]] = icmp ne i64 %{{.*}}, 0
! CHECK: omp_mapper.iterspan.body:
! CHECK: call i64 @llvm.umin.i64
! CHECK: call i64 @llvm.umax.i64
! CHECK: omp.it.cont:
! CHECK: %[[ANY_TRIPS:.*]] = or i1 false, %[[HAS_TRIPS]]
! CHECK: %[[PBEGIN:.*]] = inttoptr i64 %[[PB:.*]] to ptr
! CHECK: %[[PSIZE:.*]] = sub i64 %{{.*}}, %[[PB]]
! CHECK: br i1 %[[ANY_TRIPS]], label %omp.iterparent.push, label %omp.iterparent.cont
! CHECK: omp.iterparent.push:
! CHECK: call void @__tgt_push_mapper_component(ptr %{{.*}}, ptr %omp.arraymap.ptrcurrent, ptr %[[PBEGIN]], i64 %[[PSIZE]], i64 %{{.*}}, ptr @{{.*}})
! CHECK: omp.iterparent.cont:
! CHECK: %[[NUM:.*]] = call i64 @__tgt_mapper_num_components(ptr %{{.*}})
! CHECK: %[[BASE:.*]] = sub i64 %[[NUM]], 1
! CHECK: shl i64 %[[BASE]], 48
! CHECK: omp_mapper.iterator.cond:
! CHECK: icmp ult i64 %{{.*}}, %{{.*}}
! CHECK: omp_mapper.iterator.body:
! CHECK: %[[FIELD:.*]] = getelementptr %_QFfTs, ptr %omp.arraymap.ptrcurrent, i32 0, i32 1
! CHECK: %[[ELEM:.*]] = getelementptr inbounds [10 x i32], ptr %[[FIELD]], i64 0, i64 %{{.*}}
! CHECK: %[[SIZE:.*]] = mul i64 %{{.*}}, 4
! CHECK: add nuw i64 281474976710659, %{{.*}}
! CHECK: call void @__tgt_push_mapper_component(ptr %{{.*}}, ptr %[[FIELD]], ptr %[[ELEM]], i64 %[[SIZE]], i64 %{{.*}}, ptr null)

! This test checks lowering of OpenMP allocate Directive to LLVM IR.
! Verifies code generation for default (no align, null allocator) case.

! RUN: %flang_fc1 -emit-llvm -fopenmp %s -o - | FileCheck %s

program main
  integer :: x, y
  !$omp allocate(x, y)
end program

! CHECK: define void @_QQmain()
! CHECK: call i32 @__kmpc_global_thread_num(
! CHECK: call ptr @__kmpc_alloc(i32 {{.*}}, i64 8, ptr null)
! CHECK: call ptr @__kmpc_alloc(i32 {{.*}}, i64 8, ptr null)
! CHECK: call void @__kmpc_free(i32 {{.*}}, ptr {{.*}}, ptr null)
! CHECK: call void @__kmpc_free(i32 {{.*}}, ptr {{.*}}, ptr null)
! CHECK: ret void
! CHECK: declare noalias ptr @__kmpc_alloc(i32, i64, ptr)
! CHECK: declare void @__kmpc_free(i32, ptr, ptr)

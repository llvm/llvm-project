! This test checks lowering of OpenMP allocate Directive with align and allocator
! clauses to LLVM IR. Verifies code generation for:
!   - align(16) only (null allocator)
!   - allocator(1) only (no align)
!   - align(64) allocator(6) (both clauses, array variable)
!   - align(32) allocator(3) (both clauses, multiple variables)

! RUN: %flang_fc1 -emit-llvm %openmp_flags -fopenmp-version=51 %s -o - 2>&1 | FileCheck %s

program main
  integer :: x, y
  integer :: z(10)
  character c
  real :: r
  complex :: cmplx
  !$omp allocate(x) align(16)
  !$omp allocate(y) allocator(1)
  !$omp allocate(z) align(64) allocator(6)
  !$omp allocate(c, r, cmplx) align(32) allocator(3)
  x = 1
  y = 2
  z = x + y
  print *, "z : ", z
end program

! CHECK: call i32 @__kmpc_global_thread_num(

! CHECK: call ptr @__kmpc_aligned_alloc(i32 {{.*}}, i64 16, i64 {{.*}}, ptr null)
! CHECK: call ptr @__kmpc_alloc(i32 {{.*}}, i64 {{.*}}, ptr inttoptr (i32 1 to ptr))
! CHECK: call ptr @__kmpc_aligned_alloc(i32 {{.*}}, i64 64, i64 {{.*}}, ptr inttoptr (i32 6 to ptr))
! CHECK: call ptr @__kmpc_aligned_alloc(i32 {{.*}}, i64 32, i64 {{.*}}, ptr inttoptr (i32 3 to ptr))
! CHECK: call ptr @__kmpc_aligned_alloc(i32 {{.*}}, i64 32, i64 {{.*}}, ptr inttoptr (i32 3 to ptr))
! CHECK: call ptr @__kmpc_aligned_alloc(i32 {{.*}}, i64 32, i64 {{.*}}, ptr inttoptr (i32 3 to ptr))

! CHECK: call void @__kmpc_free(i32 {{.*}}, ptr {{.*}}, ptr inttoptr (i32 3 to ptr))
! CHECK: call void @__kmpc_free(i32 {{.*}}, ptr {{.*}}, ptr inttoptr (i32 3 to ptr))
! CHECK: call void @__kmpc_free(i32 {{.*}}, ptr {{.*}}, ptr inttoptr (i32 3 to ptr))
! CHECK: call void @__kmpc_free(i32 {{.*}}, ptr {{.*}}, ptr inttoptr (i32 6 to ptr))
! CHECK: call void @__kmpc_free(i32 {{.*}}, ptr {{.*}}, ptr inttoptr (i32 1 to ptr))
! CHECK: call void @__kmpc_free(i32 {{.*}}, ptr {{.*}}, ptr null)
! CHECK: ret void

! CHECK: declare noalias ptr @__kmpc_aligned_alloc(i32, i64, i64, ptr)
! CHECK: declare noalias ptr @__kmpc_alloc(i32, i64, ptr)
! CHECK: declare void @__kmpc_free(i32, ptr, ptr)

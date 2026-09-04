!===----------------------------------------------------------------------===!
! This directory can be used to add Integration tests involving multiple
! stages of the compiler (for eg. from Fortran to LLVM IR). It should not
! contain executable tests. We should only add tests here sparingly and only
! if there is no other way to test. Repeat this message in each test that is
! added to this directory and sub-directories.
!===----------------------------------------------------------------------===!

! This test checks lowering of the ALLOCATE clause on the OpenMP SCOPE
! construct: PRIVATE with omitted/default allocator, PRIVATE with an explicit
! allocator, FIRSTPRIVATE with allocator-backed storage, ALIGN, and an
! allocation-list order that differs from the private-list order.

! RUN: %flang_fc1 -emit-hlfir %openmp_flags -fopenmp-version=52 -o - %s 2>&1 | FileCheck %s
! RUN: %flang_fc1 -emit-llvm %openmp_flags -fopenmp-version=52 -o - %s 2>&1 | FileCheck %s --check-prefix=LLVM
! RUN: not %flang_fc1 -fsyntax-only %openmp_flags -fopenmp-version=51 %s 2>&1 | FileCheck %s --check-prefix=VERSION51

! VERSION51: error: {{.*}}ALLOCATE

subroutine scope_allocator_omitted(x, y)
  integer :: x, y
  !$omp scope private(x, y) allocate(x)
    x = 1
    y = 2
  !$omp end scope
end subroutine

! CHECK-LABEL: func.func @_QPscope_allocator_omitted
! CHECK: %[[NULL_ALLOC:.*]] = arith.constant 0 : i32
! CHECK: omp.scope allocate(%[[NULL_ALLOC]] : i32 -> %[[X:.*]]#0 : !fir.ref<i32>) allocate_private_indices([0])
! CHECK-SAME: private({{.*}} %[[X]]#0 -> %[[X_PRIVATE:.*]], {{.*}} -> %[[Y_PRIVATE:.*]] : !fir.ref<i32>, !fir.ref<i32>) {

subroutine scope_allocator_explicit(x, allocator)
  use iso_c_binding, only : c_intptr_t
  integer :: x
  integer(c_intptr_t), intent(in) :: allocator
  !$omp scope private(x) allocate(allocator(allocator): x)
    x = 1
  !$omp end scope
end subroutine

! CHECK-LABEL: func.func @_QPscope_allocator_explicit
! CHECK: %[[ALLOCATOR:.*]] = fir.load %{{.*}} : !fir.ref<i64>
! CHECK: omp.scope allocate(%[[ALLOCATOR]] : i64 -> %[[X:.*]]#0 : !fir.ref<i32>) allocate_private_indices([0])
! CHECK-SAME: private({{.*}} %[[X]]#0 -> {{.*}} : !fir.ref<i32>) {

subroutine scope_allocator_firstprivate(x)
  integer :: x
  !$omp scope firstprivate(x) allocate(x)
    x = x + 1
  !$omp end scope
end subroutine

! CHECK-LABEL: func.func @_QPscope_allocator_firstprivate
! CHECK: omp.scope allocate({{.*}} : i32 -> %[[X:.*]]#0 : !fir.ref<i32>) allocate_private_indices([0])
! CHECK-SAME: private({{.*}} %[[X]]#0 -> %[[X_PRIVATE:.*]] : !fir.ref<i32>) {
! CHECK: %[[X_DECL:.*]]:2 = hlfir.declare %[[X_PRIVATE]]
! CHECK: hlfir.assign %{{.*}} to %[[X_DECL]]#0

subroutine scope_allocator_align(x, y)
  integer :: x, y
  !$omp scope private(x, y) allocate(x) allocate(align(64): y)
    x = 1
    y = 2
  !$omp end scope
end subroutine

! CHECK-LABEL: func.func @_QPscope_allocator_align
! CHECK: omp.scope allocate(
! CHECK-SAME: allocate_alignments([0, 64]) allocate_private_indices([0, 1])
! CHECK-SAME: private(

subroutine scope_allocator_order(x, y, z, allocator_xy, allocator_z)
  use iso_c_binding, only : c_intptr_t
  integer :: x, y, z
  integer(c_intptr_t), intent(in) :: allocator_xy, allocator_z
  !$omp scope private(x, y, z) &
  !$omp& allocate(allocator(allocator_xy): y, x) &
  !$omp& allocate(allocator(allocator_z): z)
    x = 1
    y = 2
    z = 3
  !$omp end scope
end subroutine

! CHECK-LABEL: func.func @_QPscope_allocator_order
! CHECK: omp.scope allocate(
! CHECK-SAME: allocate_private_indices([1, 0, 2])
! CHECK-SAME: private(

subroutine scope_allocator_dynamic(x, allocators, n)
  use iso_c_binding, only : c_intptr_t
  integer, intent(in) :: n
  integer(c_intptr_t), intent(in) :: allocators(n)
  integer :: x, i
  do i = 1, n
    !$omp scope firstprivate(x) allocate(allocator(allocators(i)): x)
      x = x + i
    !$omp end scope
  end do
end subroutine

! CHECK-LABEL: func.func @_QPscope_allocator_dynamic
! CHECK: fir.do_loop
! CHECK: %[[DYNAMIC_ALLOCATOR:.*]] = fir.load %{{.*}} : !fir.ref<i64>
! CHECK: omp.scope allocate(%[[DYNAMIC_ALLOCATOR]] : i64 -> %[[X:.*]]#0 : !fir.ref<i32>) allocate_private_indices([0])
! CHECK-SAME: private({{.*}} %[[X]]#0 -> {{.*}} : !fir.ref<i32>) {

! LLVM-LABEL: define void @scope_allocator_dynamic_
! LLVM-NOT: call ptr @__kmpc_{{(aligned_)?}}alloc
! LLVM: omp.region.after_alloca:
! LLVM-NOT: call ptr @__kmpc_{{(aligned_)?}}alloc
! LLVM: br label %[[LOOP:[0-9]+]]
! LLVM: [[LOOP]]:
! LLVM-NOT: call ptr @__kmpc_{{(aligned_)?}}alloc
! LLVM: br i1 {{.*}}, label %[[BODY:[0-9]+]], label %[[EXIT:[0-9]+]]
! LLVM: [[BODY]]:
! LLVM: %[[DYNAMIC_ALLOCATOR:.*]] = load i64
! LLVM: %[[ALLOCATOR_HANDLE:.*]] = inttoptr i64 %[[DYNAMIC_ALLOCATOR]] to ptr
! LLVM: %[[ALLOC:.*]] = call ptr @__kmpc_alloc({{.*}}, i64 4, ptr %[[ALLOCATOR_HANDLE]])
! LLVM-NOT: call ptr @__kmpc_{{(aligned_)?}}alloc
! LLVM-NOT: call void @__kmpc_free
! LLVM: store i32 {{.*}}, ptr %[[ALLOC]]
! LLVM-NOT: call ptr @__kmpc_{{(aligned_)?}}alloc
! LLVM-NOT: call void @__kmpc_free
! LLVM: call void @__kmpc_free({{.*}}, ptr %[[ALLOC]], ptr %[[ALLOCATOR_HANDLE]])
! LLVM-NOT: call void @__kmpc_free
! LLVM: br label %[[LOOP]]
! LLVM: [[EXIT]]:
! LLVM: ret void

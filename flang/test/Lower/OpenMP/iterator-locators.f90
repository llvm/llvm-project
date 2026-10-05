! RUN: bbc -emit-hlfir -fopenmp -fopenmp-version=52 -o - %s \
! RUN:   | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %s \
! RUN:   | FileCheck %s
! RUN: %flang_fc1 -emit-llvm -fopenmp -fopenmp-version=52 -o /dev/null %s
! RUN: %flang_fc1 -emit-llvm -O1 -fopenmp -fopenmp-version=52 -o - %s \
! RUN:   | FileCheck %s --check-prefix=LLVM

! Folded and live iterator references must use ordinary locator semantics.
! All designator evaluation belongs inside the selected iterator region.

subroutine depend_component(a, m, d)
  type t
    integer :: field
  end type
  type(t) :: a(8)
  integer :: m, d
  !$omp task depend(iterator(i=1:m), in: &
  !$omp& a(1+0*i)%field, a(i/d)%field)
  !$omp end task
end
! CHECK-LABEL: func.func @_QPdepend_component(
! CHECK: %[[BASE:.*]]:2 = hlfir.declare %arg0
! CHECK-NOT: arith.divsi
! CHECK: %[[FOLDED:.*]] = omp.iterator(
! CHECK: %[[ELEM:.*]] = hlfir.designate %[[BASE]]#0 (
! CHECK: %[[ADDR:.*]] = hlfir.designate %[[ELEM]]{"field"}
! CHECK: %[[PTR:.*]] = fir.convert %[[ADDR]]
! CHECK: omp.yield(%[[PTR]] : !llvm.ptr)
! CHECK: %[[LIVE:.*]] = omp.iterator(
! CHECK: arith.divsi
! CHECK: %[[ELEM:.*]] = hlfir.designate %[[BASE]]#0 (
! CHECK: %[[ADDR:.*]] = hlfir.designate %[[ELEM]]{"field"}
! CHECK: %[[PTR:.*]] = fir.convert %[[ADDR]]
! CHECK: omp.yield(%[[PTR]] : !llvm.ptr)
! CHECK: omp.task depend(
! CHECK-SAME: %[[FOLDED]] :
! CHECK-SAME: %[[LIVE]] :

subroutine depend_vector(a, m, d)
  integer :: a(8)
  integer :: m, d
  !$omp task depend(iterator(i=1:m), in: &
  !$omp& a([1+0*i]), a([i/d]))
  !$omp end task
end
! CHECK-LABEL: func.func @_QPdepend_vector(
! CHECK: %[[BASE:.*]]:2 = hlfir.declare %arg0
! CHECK-NOT: arith.divsi
! CHECK: %[[FOLDED:.*]] = omp.iterator(
! CHECK: %[[INDEX:.*]] = fir.load {{.*}} : !fir.ref<i64>
! CHECK: %[[ADDR:.*]] = hlfir.designate %[[BASE]]#0 (%[[INDEX]])
! CHECK: %[[PTR:.*]] = fir.convert %[[ADDR]]
! CHECK: omp.yield(%[[PTR]] : !llvm.ptr)
! CHECK: %[[LIVE:.*]] = omp.iterator(
! CHECK: arith.divsi
! CHECK: %[[TMP:.*]] = hlfir.as_expr
! CHECK: %[[INDEX:.*]] = hlfir.apply %[[TMP]],
! CHECK: %[[ADDR:.*]] = hlfir.designate %[[BASE]]#0 (%[[INDEX]])
! CHECK: %[[PTR:.*]] = fir.convert %[[ADDR]]
! CHECK: hlfir.destroy %[[TMP]]
! CHECK: omp.yield(%[[PTR]] : !llvm.ptr)
! CHECK: omp.task depend(
! CHECK-SAME: %[[FOLDED]] :
! CHECK-SAME: %[[LIVE]] :

subroutine depend_section(x, m, d)
  type t
    integer :: a(8)
  end type
  type(t) :: x
  integer :: m, d
  !$omp task depend(iterator(i=1:m), in: &
  !$omp& x%a(1:2+0*i), x%a(i/d:i/d+1))
  !$omp end task
end
! CHECK-LABEL: func.func @_QPdepend_section(
! CHECK: %[[BASE:.*]]:2 = hlfir.declare %arg0
! CHECK-NOT: arith.divsi
! CHECK: %[[FOLDED:.*]] = omp.iterator(
! CHECK: %[[ADDR:.*]] = hlfir.designate %[[BASE]]#0{"a"}
! CHECK: %[[PTR:.*]] = fir.convert %[[ADDR]]
! CHECK: omp.yield(%[[PTR]] : !llvm.ptr)
! CHECK: %[[LIVE:.*]] = omp.iterator(
! CHECK: arith.divsi
! CHECK: %[[BOX:.*]] = hlfir.designate %[[BASE]]#0{"a"}
! CHECK: %[[ADDR:.*]] = fir.box_addr %[[BOX]]
! CHECK: %[[PTR:.*]] = fir.convert %[[ADDR]]
! CHECK: omp.yield(%[[PTR]] : !llvm.ptr)
! CHECK: omp.task depend(
! CHECK-SAME: %[[FOLDED]] :
! CHECK-SAME: %[[LIVE]] :

subroutine affinity_component(a, m, d)
  type t
    integer :: field
  end type
  type(t) :: a(8)
  integer :: m, d
  !$omp task affinity(iterator(i=1:m): &
  !$omp& a(1+0*i)%field, a(i/d)%field)
  !$omp end task
end
! CHECK-LABEL: func.func @_QPaffinity_component(
! CHECK: %[[BASE:.*]]:2 = hlfir.declare %arg0
! CHECK-NOT: arith.divsi
! CHECK: %[[FOLDED:.*]] = omp.iterator(
! CHECK: %[[ELEM:.*]] = hlfir.designate %[[BASE]]#0 (
! CHECK: %[[ADDR:.*]] = hlfir.designate %[[ELEM]]{"field"}
! CHECK: %[[PTR:.*]] = fir.convert %[[ADDR]]
! CHECK: %[[ENTRY:.*]] = omp.affinity_entry %[[PTR]],
! CHECK: omp.yield(%[[ENTRY]] :
! CHECK: %[[LIVE:.*]] = omp.iterator(
! CHECK: arith.divsi
! CHECK: %[[ELEM:.*]] = hlfir.designate %[[BASE]]#0 (
! CHECK: %[[ADDR:.*]] = hlfir.designate %[[ELEM]]{"field"}
! CHECK: %[[PTR:.*]] = fir.convert %[[ADDR]]
! CHECK: %[[ENTRY:.*]] = omp.affinity_entry %[[PTR]],
! CHECK: omp.yield(%[[ENTRY]] :
! CHECK: omp.task affinity(
! CHECK-SAME: %[[FOLDED]] :
! CHECK-SAME: %[[LIVE]] :

subroutine affinity_section(x, m, d)
  type t
    integer :: a(8)
  end type
  type(t) :: x
  integer :: m, d
  !$omp task affinity(iterator(i=1:m): &
  !$omp& x%a(1:2+0*i), x%a(i/d:i/d+1))
  !$omp end task
end
! CHECK-LABEL: func.func @_QPaffinity_section(
! CHECK: %[[BASE:.*]]:2 = hlfir.declare %arg0
! CHECK-NOT: arith.divsi
! CHECK: %[[FOLDED:.*]] = omp.iterator(
! CHECK: %[[ADDR:.*]] = hlfir.designate %[[BASE]]#0{"a"}
! CHECK: %[[PTR:.*]] = fir.convert %[[ADDR]]
! CHECK: %[[ENTRY:.*]] = omp.affinity_entry %[[PTR]],
! CHECK: omp.yield(%[[ENTRY]] :
! CHECK: %[[LIVE:.*]] = omp.iterator(
! CHECK: arith.divsi
! CHECK: %[[BOX:.*]] = hlfir.designate %[[BASE]]#0{"a"}
! CHECK: %[[ADDR:.*]] = fir.box_addr %[[BOX]]
! CHECK: %[[ELEMS:.*]] = fir.convert %{{.*}} : (index) -> i64
! CHECK: %[[LEN:.*]] = arith.muli %[[ELEMS]], %{{.*}} : i64
! CHECK: %[[PTR:.*]] = fir.convert %[[ADDR]]
! CHECK: %[[ENTRY:.*]] = omp.affinity_entry %[[PTR]], %[[LEN]]
! CHECK: omp.yield(%[[ENTRY]] :
! CHECK: omp.task affinity(
! CHECK-SAME: %[[FOLDED]] :
! CHECK-SAME: %[[LIVE]] :

! Scalar components occupy one element; component sections span two.
! LLVM-LABEL: define {{.*}} @affinity_component_(
! LLVM: store i64 4,
! LLVM: store i64 4,
! LLVM: call i32 @__kmpc_omp_reg_task_with_affinity(
! LLVM-LABEL: define {{.*}} @affinity_section_(
! LLVM: store i64 8,
! LLVM: call i32 @__kmpc_omp_reg_task_with_affinity(

! Empty ranges must suppress component and vector designator evaluation.

subroutine depend_component_empty(a, m, d)
  type t
    integer :: field
  end type
  type(t) :: a(8)
  integer :: m, d
  !$omp task depend(iterator(i=2:1), in: &
  !$omp& a(1+0*i)%field, a(i/d)%field)
  !$omp end task
end
! LLVM-LABEL: define {{.*}} @depend_component_empty_(
! LLVM-NOT: sdiv
! LLVM: call i32 @__kmpc_omp_task_with_deps(
! LLVM-SAME: i32 0, ptr
! LLVM-NOT: sdiv
! LLVM: ret void

subroutine depend_vector_empty(a, m, d)
  integer :: a(8)
  integer :: m, d
  !$omp task depend(iterator(i=2:1), in: &
  !$omp& a([1+0*i]), a([i/d]))
  !$omp end task
end
! LLVM-LABEL: define {{.*}} @depend_vector_empty_(
! LLVM-NOT: sdiv
! LLVM: call i32 @__kmpc_omp_task_with_deps(
! LLVM-SAME: i32 0, ptr
! LLVM-NOT: sdiv
! LLVM: ret void

subroutine depend_section_empty(x, m, d)
  type t
    integer :: a(8)
  end type
  type(t) :: x
  integer :: m, d
  !$omp task depend(iterator(i=2:1), in: &
  !$omp& x%a(1:2+0*i), x%a(i/d:i/d+1))
  !$omp end task
end
! LLVM-LABEL: define {{.*}} @depend_section_empty_(
! LLVM-NOT: sdiv
! LLVM: call i32 @__kmpc_omp_task_with_deps(
! LLVM-SAME: i32 0, ptr
! LLVM-NOT: sdiv
! LLVM: ret void

subroutine affinity_component_empty(a, m, d)
  type t
    integer :: field
  end type
  type(t) :: a(8)
  integer :: m, d
  !$omp task affinity(iterator(i=2:1): &
  !$omp& a(1+0*i)%field, a(i/d)%field)
  !$omp end task
end
! LLVM-LABEL: define {{.*}} @affinity_component_empty_(
! LLVM-NOT: sdiv
! LLVM: call i32 @__kmpc_omp_reg_task_with_affinity(
! LLVM-SAME: i32 0, ptr
! LLVM-NOT: sdiv
! LLVM: ret void

subroutine affinity_section_empty(x, m, d)
  type t
    integer :: a(8)
  end type
  type(t) :: x
  integer :: m, d
  !$omp task affinity(iterator(i=2:1): &
  !$omp& x%a(1:2+0*i), x%a(i/d:i/d+1))
  !$omp end task
end
! LLVM-LABEL: define {{.*}} @affinity_section_empty_(
! LLVM-NOT: sdiv
! LLVM: call i32 @__kmpc_omp_reg_task_with_affinity(
! LLVM-SAME: i32 0, ptr
! LLVM-NOT: sdiv
! LLVM: ret void

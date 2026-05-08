! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %s | FileCheck %s

! Tests for the iterator modifier on the depend clause across all directives
! that Flang currently supports: task, target, target enter data, target exit data,
! and target update.
! TODO: We need to add iterator test for taskwait, depobj, interop once they are
! supported.

!===============================================================================
! task
!===============================================================================

subroutine task_depend_iterator_simple()
  integer, parameter :: n = 16
  integer :: a(n)
  integer :: i

  !$omp task depend(iterator(i = 1:n), in: a(i))
  !$omp end task
end subroutine

! CHECK-LABEL: func.func @_QPtask_depend_iterator_simple()
! CHECK: %[[A:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtask_depend_iterator_simpleEa"}
! CHECK: %[[IT:.*]] = omp.iterator(%[[IV:.*]]: index) = ({{.*}} to {{.*}} step {{.*}}) {
! CHECK:   %[[IV_I32:.*]] = fir.convert %[[IV]] : (index) -> i32
! CHECK:   %[[IV_I64:.*]] = fir.convert %[[IV_I32]] : (i32) -> i64
! CHECK:   %[[SHAPE:.*]] = fir.shape %c16 : (index) -> !fir.shape<1>
! CHECK:   %[[COOR:.*]] = fir.array_coor %[[A]]#0(%[[SHAPE]]) %[[IV_I64]] : (!fir.ref<!fir.array<16xi32>>, !fir.shape<1>, i64) -> !fir.ref<i32>
! CHECK:   %[[PTR:.*]] = fir.convert %[[COOR]] : (!fir.ref<i32>) -> !llvm.ptr
! CHECK:   omp.yield(%[[PTR]] : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: omp.task depend(taskdependin -> %[[IT]] : !omp.iterated<!llvm.ptr>) {
! CHECK:   omp.terminator
! CHECK: }

subroutine task_depend_iterator_2d()
  integer, parameter :: n = 4, m = 6
  integer :: a(n, m)
  integer :: i, j

  !$omp task depend(iterator(i = 1:n, j = 1:m), inout: a(i, j))
  !$omp end task
end subroutine

! CHECK-LABEL: func.func @_QPtask_depend_iterator_2d()
! CHECK: %[[IT:.*]] = omp.iterator(%[[IV0:.*]]: index, %[[IV1:.*]]: index) = ({{.*}} to {{.*}} step {{.*}}, {{.*}} to {{.*}} step {{.*}}) {
! CHECK:   %[[IV0_I32:.*]] = fir.convert %[[IV0]] : (index) -> i32
! CHECK:   %[[IV1_I32:.*]] = fir.convert %[[IV1]] : (index) -> i32
! CHECK:   %[[IV0_I64:.*]] = fir.convert %[[IV0_I32]] : (i32) -> i64
! CHECK:   %[[IV1_I64:.*]] = fir.convert %[[IV1_I32]] : (i32) -> i64
! CHECK:   %[[SHAPE:.*]] = fir.shape %c4, %c6 : (index, index) -> !fir.shape<2>
! CHECK:   %[[COOR:.*]] = fir.array_coor %{{.*}}(%[[SHAPE]]) %[[IV0_I64]], %[[IV1_I64]] : (!fir.ref<!fir.array<4x6xi32>>, !fir.shape<2>, i64, i64) -> !fir.ref<i32>
! CHECK:   %[[PTR:.*]] = fir.convert %[[COOR]] : (!fir.ref<i32>) -> !llvm.ptr
! CHECK:   omp.yield(%[[PTR]] : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: omp.task depend(taskdependinout -> %[[IT]] : !omp.iterated<!llvm.ptr>) {

subroutine task_depend_iterator_mixed()
  integer, parameter :: n = 16
  integer :: a(n), x
  integer :: i

  !$omp task depend(iterator(i = 1:n), in: a(i)) depend(out: x)
  !$omp end task
end subroutine

! CHECK-LABEL: func.func @_QPtask_depend_iterator_mixed()
! CHECK: %[[A:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtask_depend_iterator_mixedEa"}
! CHECK: %[[X:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFtask_depend_iterator_mixedEx"}
! CHECK: %[[IT:.*]] = omp.iterator(%[[IV:.*]]: index) = ({{.*}} to {{.*}} step {{.*}}) {
! CHECK:   omp.yield(%{{.*}} : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: omp.task depend(taskdependout -> %[[X]]#0 : !fir.ref<i32>, taskdependin -> %[[IT]] : !omp.iterated<!llvm.ptr>) {

subroutine task_depend_iterator_step()
  integer, parameter :: n = 16
  integer :: a(n)
  integer :: i

  !$omp task depend(iterator(i = 1:n:2), in: a(i))
  !$omp end task
end subroutine

! CHECK-LABEL: func.func @_QPtask_depend_iterator_step()
! CHECK: %[[C1_I32:.*]] = arith.constant 1 : i32
! CHECK: %[[C16_I32:.*]] = arith.constant 16 : i32
! CHECK: %[[LB:.*]] = fir.convert %[[C1_I32]] : (i32) -> index
! CHECK: %[[UB:.*]] = fir.convert %[[C16_I32]] : (i32) -> index
! CHECK: %[[C2_I32:.*]] = arith.constant 2 : i32
! CHECK: %[[STEP:.*]] = fir.convert %[[C2_I32]] : (i32) -> index
! CHECK: %[[IT:.*]] = omp.iterator(%[[IV:.*]]: index) = (%[[LB]] to %[[UB]] step %[[STEP]]) {
! CHECK:   omp.yield(%{{.*}} : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: omp.task depend(taskdependin -> %[[IT]] : !omp.iterated<!llvm.ptr>) {

subroutine task_depend_iterator_multi_obj()
  integer, parameter :: n = 16
  integer :: a(n), b(n)
  integer :: i

  !$omp task depend(iterator(i = 1:n), inout: a(i), b(i))
  !$omp end task
end subroutine

! CHECK-LABEL: func.func @_QPtask_depend_iterator_multi_obj()
! CHECK: %[[A:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtask_depend_iterator_multi_objEa"}
! CHECK: %[[B:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtask_depend_iterator_multi_objEb"}
! CHECK: %[[IT1:.*]] = omp.iterator(%[[IV1:.*]]: index) = ({{.*}} to {{.*}} step {{.*}}) {
! CHECK:   %[[COOR1:.*]] = fir.array_coor %[[A]]#0(%{{.*}}) %{{.*}} : (!fir.ref<!fir.array<16xi32>>, !fir.shape<1>, i64) -> !fir.ref<i32>
! CHECK:   %[[PTR1:.*]] = fir.convert %[[COOR1]] : (!fir.ref<i32>) -> !llvm.ptr
! CHECK:   omp.yield(%[[PTR1]] : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: %[[IT2:.*]] = omp.iterator(%[[IV2:.*]]: index) = ({{.*}} to {{.*}} step {{.*}}) {
! CHECK:   %[[COOR2:.*]] = fir.array_coor %[[B]]#0(%{{.*}}) %{{.*}} : (!fir.ref<!fir.array<16xi32>>, !fir.shape<1>, i64) -> !fir.ref<i32>
! CHECK:   %[[PTR2:.*]] = fir.convert %[[COOR2]] : (!fir.ref<i32>) -> !llvm.ptr
! CHECK:   omp.yield(%[[PTR2]] : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: omp.task depend(taskdependinout -> %[[IT1]] : !omp.iterated<!llvm.ptr>, taskdependinout -> %[[IT2]] : !omp.iterated<!llvm.ptr>) {

! Expression-based subscript using multiple iterator variables: a((i-1)*m+j)
! maps a 2D logical iteration space onto a 1D array.
subroutine task_depend_iterator_expr_subscript()
  integer, parameter :: m = 4
  integer, parameter :: n = m * m
  integer :: a(n)
  integer :: i, j

  !$omp task depend(iterator(i = 1:m, j = 1:m), out: a((i-1)*m+j))
  !$omp end task
end subroutine

! CHECK-LABEL: func.func @_QPtask_depend_iterator_expr_subscript()
! CHECK: %[[A:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtask_depend_iterator_expr_subscriptEa"}
! CHECK: %[[IT:.*]] = omp.iterator(%[[IV0:.*]]: index, %[[IV1:.*]]: index) = ({{.*}} to {{.*}} step {{.*}}, {{.*}} to {{.*}} step {{.*}}) {
! CHECK:   %[[IV0_I32:.*]] = fir.convert %[[IV0]] : (index) -> i32
! CHECK:   %[[IV1_I32:.*]] = fir.convert %[[IV1]] : (index) -> i32
! CHECK:   %[[C1_I32:.*]] = arith.constant 1 : i32
! CHECK:   %[[SUB:.*]] = arith.subi %[[IV0_I32]], %[[C1_I32]] : i32
! CHECK:   %[[NOREASSOC:.*]] = fir.no_reassoc %[[SUB]] : i32
! CHECK:   %[[MUL:.*]] = arith.muli %{{.*}}, %[[NOREASSOC]] : i32
! CHECK:   %[[ADD:.*]] = arith.addi %[[MUL]], %[[IV1_I32]] : i32
! CHECK:   %[[IDX:.*]] = fir.convert %[[ADD]] : (i32) -> i64
! CHECK:   %[[COOR:.*]] = fir.array_coor %[[A]]#0(%{{.*}}) %[[IDX]] : (!fir.ref<!fir.array<16xi32>>, !fir.shape<1>, i64) -> !fir.ref<i32>
! CHECK:   %[[PTR:.*]] = fir.convert %[[COOR]] : (!fir.ref<i32>) -> !llvm.ptr
! CHECK:   omp.yield(%[[PTR]] : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: omp.task depend(taskdependout -> %[[IT]] : !omp.iterated<!llvm.ptr>) {

! Multiple depend clauses each with their own iterator on the same task.
subroutine task_depend_multi_iter_clauses()
  integer, parameter :: n = 8
  integer :: a(n), b(n)
  integer :: i, j

  !$omp task depend(iterator(i = 1:n), in: a(i)) &
  !$omp&     depend(iterator(j = 1:n), out: b(j))
  !$omp end task
end subroutine

! CHECK-LABEL: func.func @_QPtask_depend_multi_iter_clauses()
! CHECK: %[[A:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtask_depend_multi_iter_clausesEa"}
! CHECK: %[[B:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtask_depend_multi_iter_clausesEb"}
! CHECK: %[[IT1:.*]] = omp.iterator(%[[IV1:.*]]: index) = ({{.*}} to {{.*}} step {{.*}}) {
! CHECK:   %[[COOR1:.*]] = fir.array_coor %[[A]]#0(%{{.*}}) %{{.*}} : (!fir.ref<!fir.array<8xi32>>, !fir.shape<1>, i64) -> !fir.ref<i32>
! CHECK:   %[[PTR1:.*]] = fir.convert %[[COOR1]] : (!fir.ref<i32>) -> !llvm.ptr
! CHECK:   omp.yield(%[[PTR1]] : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: %[[IT2:.*]] = omp.iterator(%[[IV2:.*]]: index) = ({{.*}} to {{.*}} step {{.*}}) {
! CHECK:   %[[COOR2:.*]] = fir.array_coor %[[B]]#0(%{{.*}}) %{{.*}} : (!fir.ref<!fir.array<8xi32>>, !fir.shape<1>, i64) -> !fir.ref<i32>
! CHECK:   %[[PTR2:.*]] = fir.convert %[[COOR2]] : (!fir.ref<i32>) -> !llvm.ptr
! CHECK:   omp.yield(%[[PTR2]] : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: omp.task depend(taskdependin -> %[[IT1]] : !omp.iterated<!llvm.ptr>, taskdependout -> %[[IT2]] : !omp.iterated<!llvm.ptr>) {

subroutine task_depend_iterator_negative_step()
  integer, parameter :: n = 16
  integer :: a(n)
  integer :: i

  !$omp task depend(iterator(i = n:1:-1), in: a(i))
  !$omp end task
end subroutine

! CHECK-LABEL: func.func @_QPtask_depend_iterator_negative_step()
! CHECK: %[[C16_I32:.*]] = arith.constant 16 : i32
! CHECK: %[[C1_I32:.*]] = arith.constant 1 : i32
! CHECK: %[[LB:.*]] = fir.convert %[[C16_I32]] : (i32) -> index
! CHECK: %[[UB:.*]] = fir.convert %[[C1_I32]] : (i32) -> index
! CHECK: %[[CM1_I32:.*]] = arith.constant -1 : i32
! CHECK: %[[STEP:.*]] = fir.convert %[[CM1_I32]] : (i32) -> index
! CHECK: %[[IT:.*]] = omp.iterator(%[[IV:.*]]: index) = (%[[LB]] to %[[UB]] step %[[STEP]]) {
! CHECK:   omp.yield(%{{.*}} : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: omp.task depend(taskdependin -> %[[IT]] : !omp.iterated<!llvm.ptr>) {

! Mixed iterated and non-iterated objects in the same depend clause:
! a(1) does not reference the iterator IV, so it is lowered as a regular
! (non-iterated) depend var, while a(i) produces an omp.iterator.
subroutine task_depend_iterator_mixed_within_clause()
  integer, parameter :: n = 16
  integer :: a(n)
  integer :: i

  !$omp task depend(iterator(i = 2:n:2), in: a(1), a(i))
  !$omp end task
end subroutine

! CHECK-LABEL: func.func @_QPtask_depend_iterator_mixed_within_clause()
! CHECK: %[[A:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtask_depend_iterator_mixed_within_clauseEa"}
! CHECK: %[[A1:.*]] = hlfir.designate %[[A]]#0 (%{{.*}})  : (!fir.ref<!fir.array<16xi32>>, index) -> !fir.ref<i32>
! CHECK: %[[IT:.*]] = omp.iterator(%[[IV:.*]]: index) = ({{.*}} to {{.*}} step {{.*}}) {
! CHECK:   %[[COOR:.*]] = fir.array_coor %[[A]]#0(%{{.*}}) %{{.*}} : (!fir.ref<!fir.array<16xi32>>, !fir.shape<1>, i64) -> !fir.ref<i32>
! CHECK:   %[[PTR:.*]] = fir.convert %[[COOR]] : (!fir.ref<i32>) -> !llvm.ptr
! CHECK:   omp.yield(%[[PTR]] : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: omp.task depend(taskdependin -> %[[A1]] : !fir.ref<i32>, taskdependin -> %[[IT]] : !omp.iterated<!llvm.ptr>) {

!===============================================================================
! target
!===============================================================================

subroutine target_depend_iterator()
  integer, parameter :: n = 16
  integer :: a(n)
  integer :: i

  !$omp target depend(iterator(i = 1:n), in: a(i)) map(tofrom: a)
    a(1) = 10
  !$omp end target
end subroutine

! CHECK-LABEL: func.func @_QPtarget_depend_iterator()
! CHECK: %[[A:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtarget_depend_iteratorEa"}
! CHECK: %[[IT:.*]] = omp.iterator(%[[IV:.*]]: index) = ({{.*}} to {{.*}} step {{.*}}) {
! CHECK:   %[[IV_I32:.*]] = fir.convert %[[IV]] : (index) -> i32
! CHECK:   %[[IV_I64:.*]] = fir.convert %[[IV_I32]] : (i32) -> i64
! CHECK:   %[[SHAPE:.*]] = fir.shape %c16 : (index) -> !fir.shape<1>
! CHECK:   %[[COOR:.*]] = fir.array_coor %[[A]]#0(%[[SHAPE]]) %[[IV_I64]] : (!fir.ref<!fir.array<16xi32>>, !fir.shape<1>, i64) -> !fir.ref<i32>
! CHECK:   %[[PTR:.*]] = fir.convert %[[COOR]] : (!fir.ref<i32>) -> !llvm.ptr
! CHECK:   omp.yield(%[[PTR]] : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: %[[MAP:.*]] = omp.map.info var_ptr(%[[A]]#1 : {{.*}}) map_clauses(tofrom) capture(ByRef) bounds({{.*}}) -> !fir.ref<!fir.array<16xi32>> {name = "a"}
! CHECK: omp.target depend(taskdependin -> %[[IT]] : !omp.iterated<!llvm.ptr>) map_entries(%[[MAP]] -> %{{.*}} : !fir.ref<!fir.array<16xi32>>) {
! CHECK:   omp.terminator
! CHECK: }

subroutine target_depend_iterator_multi()
  integer, parameter :: n = 8
  integer :: a(n), b(n), c(n)
  integer :: i, j

  !$omp target depend(iterator(i = 1:n), inout: a(i), b(i)) &
  !$omp&       depend(iterator(j = 1:n:2), in: c(j)) &
  !$omp&       map(tofrom: a, b)
    a(1) = b(1)
  !$omp end target
end subroutine

! CHECK-LABEL: func.func @_QPtarget_depend_iterator_multi()
! CHECK: %[[A:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtarget_depend_iterator_multiEa"}
! CHECK: %[[B:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtarget_depend_iterator_multiEb"}
! CHECK: %[[C:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtarget_depend_iterator_multiEc"}
! CHECK: %[[IT1:.*]] = omp.iterator(%{{.*}}: index) = ({{.*}} to {{.*}} step {{.*}}) {
! CHECK:   %[[COOR1:.*]] = fir.array_coor %[[A]]#0(%{{.*}}) %{{.*}} : (!fir.ref<!fir.array<8xi32>>, !fir.shape<1>, i64) -> !fir.ref<i32>
! CHECK:   omp.yield(%{{.*}} : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: %[[IT2:.*]] = omp.iterator(%{{.*}}: index) = ({{.*}} to {{.*}} step {{.*}}) {
! CHECK:   %[[COOR2:.*]] = fir.array_coor %[[B]]#0(%{{.*}}) %{{.*}} : (!fir.ref<!fir.array<8xi32>>, !fir.shape<1>, i64) -> !fir.ref<i32>
! CHECK:   omp.yield(%{{.*}} : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: %[[IT3:.*]] = omp.iterator(%{{.*}}: index) = ({{.*}} to {{.*}} step {{.*}}) {
! CHECK:   %[[COOR3:.*]] = fir.array_coor %[[C]]#0(%{{.*}}) %{{.*}} : (!fir.ref<!fir.array<8xi32>>, !fir.shape<1>, i64) -> !fir.ref<i32>
! CHECK:   omp.yield(%{{.*}} : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: %[[MAP_A:.*]] = omp.map.info var_ptr(%[[A]]#1 : {{.*}}) map_clauses(tofrom) capture(ByRef) bounds({{.*}}) -> !fir.ref<!fir.array<8xi32>> {name = "a"}
! CHECK: %[[MAP_B:.*]] = omp.map.info var_ptr(%[[B]]#1 : {{.*}}) map_clauses(tofrom) capture(ByRef) bounds({{.*}}) -> !fir.ref<!fir.array<8xi32>> {name = "b"}
! CHECK: %[[MAP_C:.*]] = omp.map.info var_ptr(%[[C]]#1 : {{.*}}) map_clauses(implicit, tofrom) capture(ByRef) bounds({{.*}}) -> !fir.ref<!fir.array<8xi32>> {name = "c"}
! CHECK: omp.target depend(taskdependinout -> %[[IT1]] : !omp.iterated<!llvm.ptr>, taskdependinout -> %[[IT2]] : !omp.iterated<!llvm.ptr>, taskdependin -> %[[IT3]] : !omp.iterated<!llvm.ptr>) map_entries(%[[MAP_A]] -> %{{.*}}, %[[MAP_B]] -> %{{.*}}, %[[MAP_C]] -> %{{.*}} : !fir.ref<!fir.array<8xi32>>, !fir.ref<!fir.array<8xi32>>, !fir.ref<!fir.array<8xi32>>) {

!===============================================================================
! target enter data
!===============================================================================

subroutine target_enter_data_depend_iterator()
  integer, parameter :: n = 16
  integer :: a(n)
  integer :: i

  !$omp target enter data map(to: a) depend(iterator(i = 1:n), in: a(i))
end subroutine

! CHECK-LABEL: func.func @_QPtarget_enter_data_depend_iterator()
! CHECK: %[[A:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtarget_enter_data_depend_iteratorEa"}
! CHECK: %[[IT:.*]] = omp.iterator(%[[IV:.*]]: index) = ({{.*}} to {{.*}} step {{.*}}) {
! CHECK:   %[[IV_I32:.*]] = fir.convert %[[IV]] : (index) -> i32
! CHECK:   %[[IV_I64:.*]] = fir.convert %[[IV_I32]] : (i32) -> i64
! CHECK:   %[[SHAPE:.*]] = fir.shape %c16 : (index) -> !fir.shape<1>
! CHECK:   %[[COOR:.*]] = fir.array_coor %[[A]]#0(%[[SHAPE]]) %[[IV_I64]] : (!fir.ref<!fir.array<16xi32>>, !fir.shape<1>, i64) -> !fir.ref<i32>
! CHECK:   %[[PTR:.*]] = fir.convert %[[COOR]] : (!fir.ref<i32>) -> !llvm.ptr
! CHECK:   omp.yield(%[[PTR]] : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: %[[MAP:.*]] = omp.map.info var_ptr(%[[A]]#1 : {{.*}}) map_clauses(to) capture(ByRef) bounds({{.*}}) -> !fir.ref<!fir.array<16xi32>> {name = "a"}
! CHECK: omp.target_enter_data depend(taskdependin -> %[[IT]] : !omp.iterated<!llvm.ptr>) map_entries(%[[MAP]] : !fir.ref<!fir.array<16xi32>>)

subroutine target_enter_data_depend_iterator_expr()
  integer, parameter :: m = 4
  integer, parameter :: n = m * m
  integer :: a(n)
  integer :: i, j

  !$omp target enter data map(to: a) &
  !$omp&   depend(iterator(i = 1:m, j = 1:m), inout: a(1), a((i-1)*m+j))
end subroutine

! CHECK-LABEL: func.func @_QPtarget_enter_data_depend_iterator_expr()
! CHECK: %[[A:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtarget_enter_data_depend_iterator_exprEa"}
! CHECK: %[[A1:.*]] = hlfir.designate %[[A]]#0 (%{{.*}})  : (!fir.ref<!fir.array<16xi32>>, index) -> !fir.ref<i32>
! CHECK: %[[IT:.*]] = omp.iterator(%[[IV0:.*]]: index, %[[IV1:.*]]: index) = ({{.*}} to {{.*}} step {{.*}}, {{.*}} to {{.*}} step {{.*}}) {
! CHECK:   %[[IV0_I32:.*]] = fir.convert %[[IV0]] : (index) -> i32
! CHECK:   %[[IV1_I32:.*]] = fir.convert %[[IV1]] : (index) -> i32
! CHECK:   %[[SUB:.*]] = arith.subi %[[IV0_I32]], %{{.*}} : i32
! CHECK:   %[[NOREASSOC:.*]] = fir.no_reassoc %[[SUB]] : i32
! CHECK:   %[[MUL:.*]] = arith.muli %{{.*}}, %[[NOREASSOC]] : i32
! CHECK:   %[[ADD:.*]] = arith.addi %[[MUL]], %[[IV1_I32]] : i32
! CHECK:   %[[IDX:.*]] = fir.convert %[[ADD]] : (i32) -> i64
! CHECK:   %[[COOR:.*]] = fir.array_coor %[[A]]#0(%{{.*}}) %[[IDX]] : (!fir.ref<!fir.array<16xi32>>, !fir.shape<1>, i64) -> !fir.ref<i32>
! CHECK:   omp.yield(%{{.*}} : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: %[[MAP:.*]] = omp.map.info var_ptr(%[[A]]#1 : {{.*}}) map_clauses(to) capture(ByRef) bounds({{.*}}) -> !fir.ref<!fir.array<16xi32>> {name = "a"}
! CHECK: omp.target_enter_data depend(taskdependinout -> %[[A1]] : !fir.ref<i32>, taskdependinout -> %[[IT]] : !omp.iterated<!llvm.ptr>) map_entries(%[[MAP]] : !fir.ref<!fir.array<16xi32>>)

!===============================================================================
! target exit data
!===============================================================================

subroutine target_exit_data_depend_iterator()
  integer, parameter :: n = 16
  integer :: a(n)
  integer :: i

  !$omp target exit data map(from: a) depend(iterator(i = 1:n), out: a(i))
end subroutine

! CHECK-LABEL: func.func @_QPtarget_exit_data_depend_iterator()
! CHECK: %[[A:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtarget_exit_data_depend_iteratorEa"}
! CHECK: %[[IT:.*]] = omp.iterator(%[[IV:.*]]: index) = ({{.*}} to {{.*}} step {{.*}}) {
! CHECK:   %[[IV_I32:.*]] = fir.convert %[[IV]] : (index) -> i32
! CHECK:   %[[IV_I64:.*]] = fir.convert %[[IV_I32]] : (i32) -> i64
! CHECK:   %[[SHAPE:.*]] = fir.shape %c16 : (index) -> !fir.shape<1>
! CHECK:   %[[COOR:.*]] = fir.array_coor %[[A]]#0(%[[SHAPE]]) %[[IV_I64]] : (!fir.ref<!fir.array<16xi32>>, !fir.shape<1>, i64) -> !fir.ref<i32>
! CHECK:   %[[PTR:.*]] = fir.convert %[[COOR]] : (!fir.ref<i32>) -> !llvm.ptr
! CHECK:   omp.yield(%[[PTR]] : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: %[[MAP:.*]] = omp.map.info var_ptr(%[[A]]#1 : {{.*}}) map_clauses(from) capture(ByRef) bounds({{.*}}) -> !fir.ref<!fir.array<16xi32>> {name = "a"}
! CHECK: omp.target_exit_data depend(taskdependout -> %[[IT]] : !omp.iterated<!llvm.ptr>) map_entries(%[[MAP]] : !fir.ref<!fir.array<16xi32>>)

subroutine target_exit_data_depend_iterator_multi()
  integer, parameter :: n = 16
  integer :: a(n), b(n)
  integer :: i

  !$omp target exit data map(from: a, b) &
  !$omp&   depend(iterator(i = n:1:-1), out: a(i), b(i))
end subroutine

! CHECK-LABEL: func.func @_QPtarget_exit_data_depend_iterator_multi()
! CHECK: %[[A:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtarget_exit_data_depend_iterator_multiEa"}
! CHECK: %[[B:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtarget_exit_data_depend_iterator_multiEb"}
! CHECK: %[[C16_I32:.*]] = arith.constant 16 : i32
! CHECK: %[[C1_I32:.*]] = arith.constant 1 : i32
! CHECK: %[[LB:.*]] = fir.convert %[[C16_I32]] : (i32) -> index
! CHECK: %[[UB:.*]] = fir.convert %[[C1_I32]] : (i32) -> index
! CHECK: %[[CM1_I32:.*]] = arith.constant -1 : i32
! CHECK: %[[STEP:.*]] = fir.convert %[[CM1_I32]] : (i32) -> index
! CHECK: %[[IT1:.*]] = omp.iterator(%{{.*}}: index) = (%[[LB]] to %[[UB]] step %[[STEP]]) {
! CHECK:   %[[COOR1:.*]] = fir.array_coor %[[A]]#0(%{{.*}}) %{{.*}} : (!fir.ref<!fir.array<16xi32>>, !fir.shape<1>, i64) -> !fir.ref<i32>
! CHECK:   omp.yield(%{{.*}} : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: %[[IT2:.*]] = omp.iterator(%{{.*}}: index) = (%[[LB]] to %[[UB]] step %[[STEP]]) {
! CHECK:   %[[COOR2:.*]] = fir.array_coor %[[B]]#0(%{{.*}}) %{{.*}} : (!fir.ref<!fir.array<16xi32>>, !fir.shape<1>, i64) -> !fir.ref<i32>
! CHECK:   omp.yield(%{{.*}} : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: %[[MAP_A:.*]] = omp.map.info var_ptr(%[[A]]#1 : {{.*}}) map_clauses(from) capture(ByRef) bounds({{.*}}) -> !fir.ref<!fir.array<16xi32>> {name = "a"}
! CHECK: %[[MAP_B:.*]] = omp.map.info var_ptr(%[[B]]#1 : {{.*}}) map_clauses(from) capture(ByRef) bounds({{.*}}) -> !fir.ref<!fir.array<16xi32>> {name = "b"}
! CHECK: omp.target_exit_data depend(taskdependout -> %[[IT1]] : !omp.iterated<!llvm.ptr>, taskdependout -> %[[IT2]] : !omp.iterated<!llvm.ptr>) map_entries(%[[MAP_A]], %[[MAP_B]] : !fir.ref<!fir.array<16xi32>>, !fir.ref<!fir.array<16xi32>>)

!===============================================================================
! target update
!===============================================================================

subroutine target_update_depend_iterator()
  integer, parameter :: n = 16
  integer :: a(n)
  integer :: i

  !$omp target update to(a) depend(iterator(i = 1:n), in: a(i))
end subroutine

! CHECK-LABEL: func.func @_QPtarget_update_depend_iterator()
! CHECK: %[[A:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtarget_update_depend_iteratorEa"}
! CHECK: %[[IT:.*]] = omp.iterator(%[[IV:.*]]: index) = ({{.*}} to {{.*}} step {{.*}}) {
! CHECK:   %[[IV_I32:.*]] = fir.convert %[[IV]] : (index) -> i32
! CHECK:   %[[IV_I64:.*]] = fir.convert %[[IV_I32]] : (i32) -> i64
! CHECK:   %[[SHAPE:.*]] = fir.shape %c16 : (index) -> !fir.shape<1>
! CHECK:   %[[COOR:.*]] = fir.array_coor %[[A]]#0(%[[SHAPE]]) %[[IV_I64]] : (!fir.ref<!fir.array<16xi32>>, !fir.shape<1>, i64) -> !fir.ref<i32>
! CHECK:   %[[PTR:.*]] = fir.convert %[[COOR]] : (!fir.ref<i32>) -> !llvm.ptr
! CHECK:   omp.yield(%[[PTR]] : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: %[[MAP:.*]] = omp.map.info var_ptr(%[[A]]#1 : {{.*}}) map_clauses(to) capture(ByRef) bounds({{.*}}) -> !fir.ref<!fir.array<16xi32>> {name = "a"}
! CHECK: omp.target_update depend(taskdependin -> %[[IT]] : !omp.iterated<!llvm.ptr>) map_entries(%[[MAP]] : !fir.ref<!fir.array<16xi32>>)

! Two separate depend(iterator) clauses with different IVs and depend kinds,
! plus a non-iterated depend.
subroutine target_update_depend_iterator_multi()
  integer, parameter :: n = 8
  integer :: a(n), b(n), x
  integer :: i, j

  !$omp target update to(a) from(b) &
  !$omp&   depend(iterator(i = 1:n), in: a(i)) &
  !$omp&   depend(iterator(j = 1:n:2), out: b(j)) &
  !$omp&   depend(inout: x)
end subroutine

! CHECK-LABEL: func.func @_QPtarget_update_depend_iterator_multi()
! CHECK: %[[A:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtarget_update_depend_iterator_multiEa"}
! CHECK: %[[B:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtarget_update_depend_iterator_multiEb"}
! CHECK: %[[X:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFtarget_update_depend_iterator_multiEx"}
! CHECK: %[[IT1:.*]] = omp.iterator(%{{.*}}: index) = ({{.*}} to {{.*}} step {{.*}}) {
! CHECK:   %[[COOR1:.*]] = fir.array_coor %[[A]]#0(%{{.*}}) %{{.*}} : (!fir.ref<!fir.array<8xi32>>, !fir.shape<1>, i64) -> !fir.ref<i32>
! CHECK:   omp.yield(%{{.*}} : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: %[[IT2:.*]] = omp.iterator(%{{.*}}: index) = ({{.*}} to {{.*}} step {{.*}}) {
! CHECK:   %[[COOR2:.*]] = fir.array_coor %[[B]]#0(%{{.*}}) %{{.*}} : (!fir.ref<!fir.array<8xi32>>, !fir.shape<1>, i64) -> !fir.ref<i32>
! CHECK:   omp.yield(%{{.*}} : !llvm.ptr)
! CHECK: } -> !omp.iterated<!llvm.ptr>
! CHECK: %[[MAP_A:.*]] = omp.map.info var_ptr(%[[A]]#1 : {{.*}}) map_clauses(to) capture(ByRef) bounds({{.*}}) -> !fir.ref<!fir.array<8xi32>> {name = "a"}
! CHECK: %[[MAP_B:.*]] = omp.map.info var_ptr(%[[B]]#1 : {{.*}}) map_clauses(from) capture(ByRef) bounds({{.*}}) -> !fir.ref<!fir.array<8xi32>> {name = "b"}
! CHECK: omp.target_update depend(taskdependinout -> %[[X]]#0 : !fir.ref<i32>, taskdependin -> %[[IT1]] : !omp.iterated<!llvm.ptr>, taskdependout -> %[[IT2]] : !omp.iterated<!llvm.ptr>) map_entries(%[[MAP_A]], %[[MAP_B]] : !fir.ref<!fir.array<8xi32>>, !fir.ref<!fir.array<8xi32>>)

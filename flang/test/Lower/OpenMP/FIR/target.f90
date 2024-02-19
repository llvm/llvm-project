!RUN: %flang_fc1 -emit-fir -flang-deprecated-no-hlfir -fopenmp %s -o - | FileCheck %s

!===============================================================================
! Target_Enter Simple
!===============================================================================

!CHECK-LABEL: func.func @_QPomp_target_enter_simple() {
subroutine omp_target_enter_simple
   integer :: a(1024)
   !CHECK: %[[BOUNDS:.*]] = omp.bounds   lower_bound({{.*}}) upper_bound({{.*}}) extent({{.*}}) stride({{.*}}) start_idx({{.*}})
   !CHECK: %[[MAP:.*]] = omp.map_info var_ptr({{.*}})   map_clauses(to) capture(ByRef) bounds(%[[BOUNDS]]) -> !fir.ref<!fir.array<1024xi32>> {name = "a"}
   !CHECK: omp.target_enter_data   map_entries(%[[MAP]] : !fir.ref<!fir.array<1024xi32>>)
   !$omp target enter data map(to: a)
end subroutine omp_target_enter_simple

!===============================================================================
! Target_Enter Map types
!===============================================================================

!CHECK-LABEL: func.func @_QPomp_target_enter_mt() {
subroutine omp_target_enter_mt
   integer :: a(1024)
   integer :: b(1024)
   integer :: c(1024)
   integer :: d(1024)
   !CHECK: %[[BOUNDS_0:.*]] = omp.bounds   lower_bound({{.*}}) upper_bound({{.*}}) extent({{.*}}) stride({{.*}}) start_idx({{.*}})
   !CHECK: %[[MAP_0:.*]] = omp.map_info var_ptr({{.*}})   map_clauses(to) capture(ByRef) bounds(%[[BOUNDS_0]]) -> !fir.ref<!fir.array<1024xi32>> {name = "a"}
   !CHECK: %[[BOUNDS_1:.*]] = omp.bounds   lower_bound({{.*}}) upper_bound({{.*}}) extent({{.*}}) stride({{.*}}) start_idx({{.*}})
   !CHECK: %[[MAP_1:.*]] = omp.map_info var_ptr(%{{.*}})  map_clauses(to) capture(ByRef) bounds(%[[BOUNDS_1]]) -> !fir.ref<!fir.array<1024xi32>> {name = "b"}
   !CHECK: %[[BOUNDS_2:.*]] = omp.bounds   lower_bound({{.*}}) upper_bound({{.*}}) extent({{.*}}) stride({{.*}}) start_idx({{.*}})
   !CHECK: %[[MAP_2:.*]] = omp.map_info var_ptr({{.*}})   map_clauses(always, exit_release_or_enter_alloc) capture(ByRef) bounds(%[[BOUNDS_2]]) -> !fir.ref<!fir.array<1024xi32>> {name = "c"}
   !CHECK: %[[BOUNDS_3:.*]] = omp.bounds   lower_bound({{.*}}) upper_bound({{.*}}) extent({{.*}}) stride({{.*}}) start_idx({{.*}})
   !CHECK: %[[MAP_3:.*]] = omp.map_info var_ptr({{.*}})   map_clauses(to) capture(ByRef) bounds(%[[BOUNDS_3]]) -> !fir.ref<!fir.array<1024xi32>> {name = "d"}
   !CHECK: omp.target_enter_data   map_entries(%[[MAP_0]], %[[MAP_1]], %[[MAP_2]], %[[MAP_3]] : !fir.ref<!fir.array<1024xi32>>, !fir.ref<!fir.array<1024xi32>>, !fir.ref<!fir.array<1024xi32>>, !fir.ref<!fir.array<1024xi32>>)
   !$omp target enter data map(to: a, b) map(always, alloc: c) map(to: d)
end subroutine omp_target_enter_mt

!===============================================================================
! `Nowait` clause
!===============================================================================

!CHECK-LABEL: func.func @_QPomp_target_enter_nowait() {
subroutine omp_target_enter_nowait
   integer :: a(1024)
   !CHECK: %[[BOUNDS:.*]] = omp.bounds   lower_bound({{.*}}) upper_bound({{.*}}) extent({{.*}}) stride({{.*}}) start_idx({{.*}})
   !CHECK: %[[MAP:.*]] = omp.map_info var_ptr({{.*}})   map_clauses(to) capture(ByRef) bounds(%[[BOUNDS]]) -> !fir.ref<!fir.array<1024xi32>> {name = "a"}
   !CHECK: omp.target_enter_data  nowait map_entries(%[[MAP]] : !fir.ref<!fir.array<1024xi32>>)
   !$omp target enter data map(to: a) nowait
end subroutine omp_target_enter_nowait

!===============================================================================
! `if` clause
!===============================================================================

!CHECK-LABEL: func.func @_QPomp_target_enter_if() {
subroutine omp_target_enter_if
   integer :: a(1024)
   integer :: i
   i = 5
   !CHECK: %[[VAL_3:.*]] = fir.load %[[VAL_1:.*]] : !fir.ref<i32>
   !CHECK: %[[VAL_4:.*]] = arith.constant 10 : i32
   !CHECK: %[[VAL_5:.*]] = arith.cmpi slt, %[[VAL_3]], %[[VAL_4]] : i32
   !CHECK: %[[BOUNDS:.*]] = omp.bounds   lower_bound({{.*}}) upper_bound({{.*}}) extent({{.*}}) stride({{.*}}) start_idx({{.*}})
   !CHECK: %[[MAP:.*]] = omp.map_info var_ptr({{.*}})   map_clauses(to) capture(ByRef) bounds(%[[BOUNDS]]) -> !fir.ref<!fir.array<1024xi32>> {name = "a"}
   !CHECK: omp.target_enter_data   if(%[[VAL_5]] : i1) map_entries(%[[MAP]] : !fir.ref<!fir.array<1024xi32>>)
   !$omp target enter data if(i<10) map(to: a)
end subroutine omp_target_enter_if

!===============================================================================
! `device` clause
!===============================================================================

!CHECK-LABEL: func.func @_QPomp_target_enter_device() {
subroutine omp_target_enter_device
   integer :: a(1024)
   !CHECK: %[[VAL_1:.*]] = arith.constant 2 : i32
   !CHECK: %[[BOUNDS:.*]] = omp.bounds   lower_bound({{.*}}) upper_bound({{.*}}) extent({{.*}}) stride({{.*}}) start_idx({{.*}})
   !CHECK: %[[MAP:.*]] = omp.map_info var_ptr({{.*}})   map_clauses(to) capture(ByRef) bounds(%[[BOUNDS]]) -> !fir.ref<!fir.array<1024xi32>> {name = "a"}
   !CHECK: omp.target_enter_data   device(%[[VAL_1]] : i32) map_entries(%[[MAP]] : !fir.ref<!fir.array<1024xi32>>)
   !$omp target enter data map(to: a) device(2)
end subroutine omp_target_enter_device

!===============================================================================
! Target_Exit Simple
!===============================================================================

!CHECK-LABEL: func.func @_QPomp_target_exit_simple() {
subroutine omp_target_exit_simple
   integer :: a(1024)
   !CHECK: %[[BOUNDS:.*]] = omp.bounds   lower_bound({{.*}}) upper_bound({{.*}}) extent({{.*}}) stride({{.*}}) start_idx({{.*}})
   !CHECK: %[[MAP:.*]] = omp.map_info var_ptr({{.*}})   map_clauses(from) capture(ByRef) bounds(%[[BOUNDS]]) -> !fir.ref<!fir.array<1024xi32>> {name = "a"}
   !CHECK: omp.target_exit_data   map_entries(%[[MAP]] : !fir.ref<!fir.array<1024xi32>>)
   !$omp target exit data map(from: a)
end subroutine omp_target_exit_simple

!===============================================================================
! Target_Exit Map types
!===============================================================================

!CHECK-LABEL: func.func @_QPomp_target_exit_mt() {
subroutine omp_target_exit_mt
   integer :: a(1024)
   integer :: b(1024)
   integer :: c(1024)
   integer :: d(1024)
   integer :: e(1024)
   !CHECK: %[[BOUNDS_0:.*]] = omp.bounds   lower_bound({{.*}}) upper_bound({{.*}}) extent({{.*}}) stride({{.*}}) start_idx({{.*}})
   !CHECK: %[[MAP_0:.*]] = omp.map_info var_ptr({{.*}})   map_clauses(from) capture(ByRef) bounds(%[[BOUNDS_0]]) -> !fir.ref<!fir.array<1024xi32>> {name = "a"}
   !CHECK: %[[BOUNDS_1:.*]] = omp.bounds   lower_bound({{.*}}) upper_bound({{.*}}) extent({{.*}}) stride({{.*}}) start_idx({{.*}})
   !CHECK: %[[MAP_1:.*]] = omp.map_info var_ptr({{.*}})   map_clauses(from) capture(ByRef) bounds(%[[BOUNDS_1]]) -> !fir.ref<!fir.array<1024xi32>> {name = "b"}
   !CHECK: %[[BOUNDS_2:.*]] = omp.bounds   lower_bound({{.*}}) upper_bound({{.*}}) extent({{.*}}) stride({{.*}}) start_idx({{.*}})
   !CHECK: %[[MAP_2:.*]] = omp.map_info var_ptr({{.*}})   map_clauses(exit_release_or_enter_alloc) capture(ByRef) bounds(%[[BOUNDS_2]]) -> !fir.ref<!fir.array<1024xi32>> {name = "c"}
   !CHECK: %[[BOUNDS_3:.*]] = omp.bounds   lower_bound({{.*}}) upper_bound({{.*}}) extent({{.*}}) stride({{.*}}) start_idx({{.*}})
   !CHECK: %[[MAP_3:.*]] = omp.map_info var_ptr({{.*}})   map_clauses(always, delete) capture(ByRef) bounds(%[[BOUNDS_3]]) -> !fir.ref<!fir.array<1024xi32>> {name = "d"}
   !CHECK: %[[BOUNDS_4:.*]] = omp.bounds   lower_bound({{.*}}) upper_bound({{.*}}) extent({{.*}}) stride({{.*}}) start_idx({{.*}})
   !CHECK: %[[MAP_4:.*]] = omp.map_info var_ptr({{.*}})   map_clauses(from) capture(ByRef) bounds(%[[BOUNDS_4]]) -> !fir.ref<!fir.array<1024xi32>> {name = "e"}
   !CHECK: omp.target_exit_data map_entries(%[[MAP_0]], %[[MAP_1]], %[[MAP_2]], %[[MAP_3]], %[[MAP_4]] : !fir.ref<!fir.array<1024xi32>>, !fir.ref<!fir.array<1024xi32>>, !fir.ref<!fir.array<1024xi32>>, !fir.ref<!fir.array<1024xi32>>, !fir.ref<!fir.array<1024xi32>>)
   !$omp target exit data map(from: a,b) map(release: c) map(always, delete: d) map(from: e)
end subroutine omp_target_exit_mt

!===============================================================================
! `device` clause
!===============================================================================

!CHECK-LABEL: func.func @_QPomp_target_exit_device() {
subroutine omp_target_exit_device
   integer :: a(1024)
   integer :: d
   !CHECK: %[[VAL_2:.*]] = fir.load %[[VAL_1:.*]] : !fir.ref<i32>
   !CHECK: %[[BOUNDS:.*]] = omp.bounds   lower_bound({{.*}}) upper_bound({{.*}}) extent({{.*}}) stride({{.*}}) start_idx({{.*}})
   !CHECK: %[[MAP:.*]] = omp.map_info var_ptr({{.*}})   map_clauses(from) capture(ByRef) bounds(%[[BOUNDS]]) -> !fir.ref<!fir.array<1024xi32>> {name = "a"}
   !CHECK: omp.target_exit_data   device(%[[VAL_2]] : i32) map_entries(%[[MAP]] : !fir.ref<!fir.array<1024xi32>>)
   !$omp target exit data map(from: a) device(d)
end subroutine omp_target_exit_device

!===============================================================================
! Target_Update `to` clause
!===============================================================================

subroutine omp_target_update_to
   integer :: a(1024)

   !CHECK-DAG: %[[A_ALLOC:.*]] = fir.alloca !fir.array<1024xi32> {bindc_name = "a", uniq_name = "_QFomp_target_update_toEa"}
   !CHECK-DAG: %[[BOUNDS:.*]] = omp.bounds

   !CHECK: %[[TO_MAP:.*]] = omp.map_info var_ptr(%[[A_ALLOC]] : !fir.ref<!fir.array<1024xi32>>, !fir.array<1024xi32>)
   !CHECK-SAME: map_clauses(to) capture(ByRef)
   !CHECK-SAME: bounds(%[[BOUNDS]]) -> !fir.ref<!fir.array<1024xi32>> {name = "a"}

   !CHECK: omp.target_update_data
   !CHECK-SAME: motion_entries(%[[TO_MAP]] : !fir.ref<!fir.array<1024xi32>>)
   !$omp target update to(a)
end subroutine omp_target_update_to

!===============================================================================
! Target_Update `from` clause
!===============================================================================

subroutine omp_target_update_from
   integer :: a(1024)

   !CHECK-DAG: %[[A_ALLOC:.*]] = fir.alloca !fir.array<1024xi32> {bindc_name = "a", uniq_name = "_QFomp_target_update_fromEa"}
   !CHECK-DAG: %[[BOUNDS:.*]] = omp.bounds

   !CHECK: %[[FROM_MAP:.*]] = omp.map_info var_ptr(%[[A_ALLOC]] : !fir.ref<!fir.array<1024xi32>>, !fir.array<1024xi32>)
   !CHECK-SAME: map_clauses(from) capture(ByRef)
   !CHECK-SAME: bounds(%[[BOUNDS]]) -> !fir.ref<!fir.array<1024xi32>> {name = "a"}

   !CHECK: omp.target_update_data
   !CHECK-SAME: motion_entries(%[[FROM_MAP]] : !fir.ref<!fir.array<1024xi32>>)
   !$omp target update from(a)
end subroutine omp_target_update_from

!===============================================================================
! Target_Update `if` clause
!===============================================================================

subroutine omp_target_update_if
   integer :: a(1024)
   logical :: i

   !CHECK-DAG: %[[A_ALLOC:.*]] = fir.alloca
   !CHECK-DAG: %[[BOUNDS:.*]] = omp.bounds
   !CHECK-DAG: %[[COND:.*]] = fir.convert %{{.*}} : (!fir.logical<4>) -> i1

   !CHECK: %[[TO_MAP:.*]] = omp.map_info

   !CHECK: omp.target_update_data if(%[[COND]] : i1)
   !CHECK-SAME: motion_entries(%[[TO_MAP]] : !fir.ref<!fir.array<1024xi32>>)
   !$omp target update to(a) if(i)
end subroutine omp_target_update_if

!===============================================================================
! Target_Update `device` clause
!===============================================================================

subroutine omp_target_update_device
   integer :: a(1024)

   !CHECK-DAG: %[[A_ALLOC:.*]] = fir.alloca
   !CHECK-DAG: %[[BOUNDS:.*]] = omp.bounds
   !CHECK-DAG: %[[DEVICE:.*]] = arith.constant 1 : i32

   !CHECK: %[[TO_MAP:.*]] = omp.map_info

   !CHECK: omp.target_update_data
   !CHECK-SAME: device(%[[DEVICE]] : i32)
   !CHECK-SAME: motion_entries(%[[TO_MAP]] : !fir.ref<!fir.array<1024xi32>>)
   !$omp target update to(a) device(1)
end subroutine omp_target_update_device

!===============================================================================
! Target_Update `nowait` clause
!===============================================================================

subroutine omp_target_update_nowait
   integer :: a(1024)

   !CHECK-DAG: %[[A_ALLOC:.*]] = fir.alloca
   !CHECK-DAG: %[[BOUNDS:.*]] = omp.bounds

   !CHECK: %[[TO_MAP:.*]] = omp.map_info

   !CHECK: omp.target_update_data
   !CHECK-SAME: nowait
   !CHECK-SAME: motion_entries(%[[TO_MAP]] : !fir.ref<!fir.array<1024xi32>>)
   !$omp target update to(a) nowait
end subroutine omp_target_update_nowait

!===============================================================================
! Target_Data with region
!===============================================================================

!CHECK-LABEL: func.func @_QPomp_target_data() {
subroutine omp_target_data
   !CHECK: %[[VAL_0:.*]] = fir.alloca !fir.array<1024xi32> {bindc_name = "a", uniq_name = "_QFomp_target_dataEa"}
   integer :: a(1024)
   !CHECK: %[[BOUNDS:.*]] = omp.bounds   lower_bound({{.*}}) upper_bound({{.*}}) extent({{.*}}) stride({{.*}}) start_idx({{.*}})
   !CHECK: %[[MAP:.*]] = omp.map_info var_ptr(%[[VAL_0]] : !fir.ref<!fir.array<1024xi32>>, !fir.array<1024xi32>)   map_clauses(tofrom) capture(ByRef) bounds(%[[BOUNDS]]) -> !fir.ref<!fir.array<1024xi32>> {name = "a"}
   !CHECK: omp.target_data   map_entries(%[[MAP]] : !fir.ref<!fir.array<1024xi32>>) {
   !$omp target data map(tofrom: a)
      !CHECK: %[[VAL_1:.*]] = arith.constant 10 : i32
      !CHECK: %[[VAL_2:.*]] = arith.constant 1 : i64
      !CHECK: %[[VAL_3:.*]] = arith.constant 1 : i64
      !CHECK: %[[VAL_4:.*]] = arith.subi %[[VAL_2]], %[[VAL_3]] : i64
      !CHECK: %[[VAL_5:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_4]] : (!fir.ref<!fir.array<1024xi32>>, i64) -> !fir.ref<i32>
      !CHECK: fir.store %[[VAL_1]] to %[[VAL_5]] : !fir.ref<i32>
      a(1) = 10
   !CHECK: omp.terminator
   !$omp end target data
   !CHECK: }
end subroutine omp_target_data

!CHECK-LABEL: func.func @_QPomp_target_data_mt
subroutine omp_target_data_mt
   integer :: a(1024)
   integer :: b(1024)
   !CHECK: %[[VAR_A:.*]] = fir.alloca !fir.array<1024xi32> {bindc_name = "a", uniq_name = "_QFomp_target_data_mtEa"}
   !CHECK: %[[VAR_B:.*]] = fir.alloca !fir.array<1024xi32> {bindc_name = "b", uniq_name = "_QFomp_target_data_mtEb"}
   !CHECK: %[[BOUNDS_A:.*]] = omp.bounds   lower_bound({{.*}}) upper_bound({{.*}}) extent({{.*}}) stride({{.*}}) start_idx({{.*}})
   !CHECK: %[[MAP_A:.*]] = omp.map_info var_ptr(%[[VAR_A]] : !fir.ref<!fir.array<1024xi32>>, !fir.array<1024xi32>)   map_clauses(tofrom) capture(ByRef) bounds(%[[BOUNDS_A]]) -> !fir.ref<!fir.array<1024xi32>> {name = "a"}
   !CHECK: omp.target_data   map_entries(%[[MAP_A]] : !fir.ref<!fir.array<1024xi32>>) {
   !$omp target data map(a)
   !CHECK: omp.terminator
   !$omp end target data
   !CHECK: }
   !CHECK: %[[BOUNDS_B:.*]] = omp.bounds   lower_bound({{.*}}) upper_bound({{.*}}) extent({{.*}}) stride({{.*}}) start_idx({{.*}})
   !CHECK: %[[MAP_B:.*]] = omp.map_info var_ptr(%[[VAR_B]] : !fir.ref<!fir.array<1024xi32>>, !fir.array<1024xi32>)   map_clauses(always, from) capture(ByRef) bounds(%[[BOUNDS_B]]) -> !fir.ref<!fir.array<1024xi32>> {name = "b"}
   !CHECK: omp.target_data   map_entries(%[[MAP_B]] : !fir.ref<!fir.array<1024xi32>>) {
   !$omp target data map(always, from : b)
   !CHECK: omp.terminator
   !$omp end target data
   !CHECK: }
end subroutine omp_target_data_mt

!===============================================================================
! Target with region
!===============================================================================

!CHECK-LABEL: func.func @_QPomp_target() {
subroutine omp_target
   !CHECK: %[[VAL_0:.*]] = fir.alloca !fir.array<1024xi32> {bindc_name = "a", uniq_name = "_QFomp_targetEa"}
   integer :: a(1024)
   !CHECK: %[[BOUNDS:.*]] = omp.bounds   lower_bound({{.*}}) upper_bound({{.*}}) extent({{.*}}) stride({{.*}}) start_idx({{.*}})
   !CHECK: %[[MAP:.*]] = omp.map_info var_ptr(%[[VAL_0]] : !fir.ref<!fir.array<1024xi32>>, !fir.array<1024xi32>)   map_clauses(tofrom) capture(ByRef) bounds(%[[BOUNDS]]) -> !fir.ref<!fir.array<1024xi32>> {name = "a"}
   !CHECK: omp.target   map_entries(%[[MAP]] -> %[[ARG_0:.*]] : !fir.ref<!fir.array<1024xi32>>) {
   !CHECK: ^bb0(%[[ARG_0]]: !fir.ref<!fir.array<1024xi32>>):
   !$omp target map(tofrom: a)
      !CHECK: %[[VAL_1:.*]] = arith.constant 10 : i32
      !CHECK: %[[VAL_2:.*]] = arith.constant 1 : i64
      !CHECK: %[[VAL_3:.*]] = arith.constant 1 : i64
      !CHECK: %[[VAL_4:.*]] = arith.subi %[[VAL_2]], %[[VAL_3]] : i64
      !CHECK: %[[VAL_5:.*]] = fir.coordinate_of %[[ARG_0]], %[[VAL_4]] : (!fir.ref<!fir.array<1024xi32>>, i64) -> !fir.ref<i32>
      !CHECK: fir.store %[[VAL_1]] to %[[VAL_5]] : !fir.ref<i32>
      a(1) = 10
   !CHECK: omp.terminator
   !$omp end target
   !CHECK: }
end subroutine omp_target

!===============================================================================
! Target implicit capture
!===============================================================================

!CHECK-LABEL: func.func @_QPomp_target_implicit() {
subroutine omp_target_implicit
   !CHECK: %[[VAL_0:.*]] = fir.alloca !fir.array<1024xi32> {bindc_name = "a", uniq_name = "_QFomp_target_implicitEa"}
   integer :: a(1024)
   !CHECK: %[[MAP:.*]] = omp.map_info var_ptr(%[[VAL_0]] : !fir.ref<!fir.array<1024xi32>>, !fir.array<1024xi32>)   map_clauses(implicit, tofrom) capture(ByRef) bounds(%{{.*}}) -> !fir.ref<!fir.array<1024xi32>> {name = "a"}
   !CHECK: omp.target   map_entries(%[[MAP]] -> %[[ARG_0:.*]] : !fir.ref<!fir.array<1024xi32>>) {
   !CHECK: ^bb0(%[[ARG_0]]: !fir.ref<!fir.array<1024xi32>>):
   !$omp target
      !CHECK: %[[VAL_5:.*]] = fir.coordinate_of %[[ARG_0]], %{{.*}} : (!fir.ref<!fir.array<1024xi32>>, i64) -> !fir.ref<i32>
      a(1) = 10
   !CHECK: omp.terminator
   !$omp end target
   !CHECK: }
end subroutine omp_target_implicit

!===============================================================================
! Target implicit capture nested
!===============================================================================

!CHECK-LABEL: func.func @_QPomp_target_implicit_nested() {
subroutine omp_target_implicit_nested
   integer::a, b
   !CHECK: omp.target   map_entries(%{{.*}} -> %[[ARG0:.*]], %{{.*}} -> %[[ARG1:.*]] : !fir.ref<i32>, !fir.ref<i32>) {
   !CHECK: ^bb0(%[[ARG0]]: !fir.ref<i32>, %[[ARG1]]: !fir.ref<i32>):
   !$omp target
      !CHECK: fir.store %{{.*}} to %[[ARG0]] : !fir.ref<i32>
      a = 10
      !$omp parallel
         !CHECK: fir.store %{{.*}} to %[[ARG1]] : !fir.ref<i32>
         b = 20
         !CHECK: omp.terminator
      !$omp end parallel
   !CHECK: omp.terminator
   !$omp end target
   !CHECK: }
end subroutine omp_target_implicit_nested

!===============================================================================
! Target implicit capture with bounds
!===============================================================================


!CHECK-LABEL: func.func @_QPomp_target_implicit_bounds(
!CHECK: %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "n"}) {
subroutine omp_target_implicit_bounds(n)
   !CHECK: %[[VAL_COPY:.*]] = fir.alloca i32
   !CHECK: %[[VAL_1:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
   !CHECK: fir.store %[[VAL_1]] to %[[VAL_COPY]] : !fir.ref<i32>
   !CHECK: %[[VAL_2:.*]] = fir.convert %[[VAL_1]] : (i32) -> i64
   !CHECK: %[[VAL_3:.*]] = fir.convert %[[VAL_2]] : (i64) -> index
   !CHECK: %[[VAL_4:.*]] = arith.constant 0 : index
   !CHECK: %[[VAL_5:.*]] = arith.cmpi sgt, %[[VAL_3]], %[[VAL_4]] : index
   !CHECK: %[[VAL_6:.*]] = arith.select %[[VAL_5]], %[[VAL_3]], %[[VAL_4]] : index
   !CHECK: %[[VAL_7:.*]] = arith.constant 1024 : i64
   !CHECK: %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (i64) -> index
   !CHECK: %[[VAL_9:.*]] = arith.constant 0 : index
   !CHECK: %[[VAL_10:.*]] = arith.cmpi sgt, %[[VAL_8]], %[[VAL_9]] : index
   !CHECK: %[[VAL_11:.*]] = arith.select %[[VAL_10]], %[[VAL_8]], %[[VAL_9]] : index
   !CHECK: %[[VAL_12:.*]] = fir.alloca !fir.array<?x1024xi32>, %[[VAL_6]] {bindc_name = "a", uniq_name = "_QFomp_target_implicit_boundsEa"}
   integer :: n
   integer :: a(n, 1024)
   !CHECK: %[[VAL_13:.*]] = arith.constant 1 : index
   !CHECK: %[[VAL_14:.*]] = arith.constant 0 : index
   !CHECK: %[[VAL_15:.*]] = arith.subi %[[VAL_6]], %[[VAL_13]] : index
   !CHECK: %[[VAL_16:.*]] = omp.bounds lower_bound(%[[VAL_14]] : index) upper_bound(%[[VAL_15]] : index) extent(%[[VAL_6]] : index) stride(%[[VAL_13]] : index) start_idx(%[[VAL_13]] : index)
   !CHECK: %[[VAL_17:.*]] = arith.constant 0 : index
   !CHECK: %[[VAL_18:.*]] = arith.subi %[[VAL_11]], %[[VAL_13]] : index
   !CHECK: %[[VAL_19:.*]] = omp.bounds lower_bound(%[[VAL_17]] : index) upper_bound(%[[VAL_18]] : index) extent(%[[VAL_11]] : index) stride(%[[VAL_13]] : index) start_idx(%[[VAL_13]] : index)
   !CHECK: %[[VAL_20:.*]] = omp.map_info var_ptr(%[[VAL_12]] : !fir.ref<!fir.array<?x1024xi32>>, !fir.array<?x1024xi32>) map_clauses(implicit, tofrom) capture(ByRef) bounds(%[[VAL_16]], %[[VAL_19]]) -> !fir.ref<!fir.array<?x1024xi32>> {name = "a"}
   !CHECK: %[[VAL_21:.*]] = omp.map_info var_ptr(%[[VAL_COPY]] : !fir.ref<i32>, i32) map_clauses(implicit, exit_release_or_enter_alloc) capture(ByCopy) -> !fir.ref<i32> {name = ""}
   !CHECK: omp.target map_entries(%[[VAL_20]] -> %[[VAL_22:.*]], %[[VAL_21]] -> %[[VAL_23:.*]] : !fir.ref<!fir.array<?x1024xi32>>, !fir.ref<i32>) {
   !CHECK: ^bb0(%[[VAL_22]]: !fir.ref<!fir.array<?x1024xi32>>, %[[VAL_23]]: !fir.ref<i32>):
   !$omp target
      !CHECK: %[[VAL_24:.*]] = fir.load %[[VAL_23]] : !fir.ref<i32>
      !CHECK: %[[VAL_25:.*]] = fir.convert %[[VAL_24]] : (i32) -> i64
      !CHECK: %[[VAL_26:.*]] = arith.constant 0 : index
      !CHECK: %[[VAL_27:.*]] = fir.convert %[[VAL_25]] : (i64) -> index
      !CHECK: %[[VAL_28:.*]] = arith.cmpi sgt, %[[VAL_27]], %[[VAL_26]] : index
      !CHECK: %[[VAL_29:.*]] = arith.select %[[VAL_28]], %[[VAL_27]], %[[VAL_26]] : index
      !CHECK: %[[VAL_30:.*]] = arith.constant 33 : i32
      !CHECK: %[[VAL_31:.*]] = fir.convert %[[VAL_22]] : (!fir.ref<!fir.array<?x1024xi32>>) -> !fir.ref<!fir.array<?xi32>>
      !CHECK: %[[VAL_32:.*]] = arith.constant 1 : index
      !CHECK: %[[VAL_33:.*]] = arith.constant 0 : index
      !CHECK: %[[VAL_34:.*]] = arith.constant 11 : i64
      !CHECK: %[[VAL_35:.*]] = fir.convert %[[VAL_34]] : (i64) -> index
      !CHECK: %[[VAL_36:.*]] = arith.subi %[[VAL_35]], %[[VAL_32]] : index
      !CHECK: %[[VAL_37:.*]] = arith.muli %[[VAL_32]], %[[VAL_36]] : index
      !CHECK: %[[VAL_38:.*]] = arith.addi %[[VAL_37]], %[[VAL_33]] : index
      !CHECK: %[[VAL_39:.*]] = arith.muli %[[VAL_32]], %[[VAL_29]] : index
      !CHECK: %[[VAL_40:.*]] = arith.constant 22 : i64
      !CHECK: %[[VAL_41:.*]] = fir.convert %[[VAL_40]] : (i64) -> index
      !CHECK: %[[VAL_42:.*]] = arith.subi %[[VAL_41]], %[[VAL_32]] : index
      !CHECK: %[[VAL_43:.*]] = arith.muli %[[VAL_39]], %[[VAL_42]] : index
      !CHECK: %[[VAL_44:.*]] = arith.addi %[[VAL_43]], %[[VAL_38]] : index
      !CHECK: %[[VAL_45:.*]] = fir.coordinate_of %[[VAL_31]], %[[VAL_44]] : (!fir.ref<!fir.array<?xi32>>, index) -> !fir.ref<i32>
      !CHECK: fir.store %[[VAL_30]] to %[[VAL_45]] : !fir.ref<i32>
      a(11, 22) = 33
      !CHECK: omp.terminator
   !$omp end target
!CHECK: }
end subroutine omp_target_implicit_bounds

!===============================================================================
! Target `thread_limit` clause
!===============================================================================

!CHECK-LABEL: func.func @_QPomp_target_thread_limit() {
subroutine omp_target_thread_limit
   integer :: a
   !CHECK: %[[VAL_1:.*]] = arith.constant 64 : i32
   !CHECK: %[[MAP:.*]] = omp.map_info var_ptr({{.*}})   map_clauses(tofrom) capture(ByRef) -> !fir.ref<i32> {name = "a"}
   !CHECK: omp.target   thread_limit(%[[VAL_1]] : i32) map_entries(%[[MAP]] -> %[[ARG_0:.*]] : !fir.ref<i32>) {
   !CHECK: ^bb0(%[[ARG_0]]: !fir.ref<i32>):
   !$omp target map(tofrom: a) thread_limit(64)
      a = 10
   !CHECK: omp.terminator
   !$omp end target
   !CHECK: }
end subroutine omp_target_thread_limit

!===============================================================================
! Target `use_device_ptr` clause
!===============================================================================

!CHECK-LABEL: func.func @_QPomp_target_device_ptr() {
subroutine omp_target_device_ptr
   use iso_c_binding, only : c_ptr, c_loc
   type(c_ptr) :: a
   integer, target :: b
   !CHECK: %[[MAP:.*]] = omp.map_info var_ptr({{.*}})   map_clauses(tofrom) capture(ByRef) -> {{.*}} {name = "a"}
   !CHECK: omp.target_data map_entries(%[[MAP]]{{.*}}
   !$omp target data map(tofrom: a) use_device_ptr(a)
   !CHECK: ^bb0(%[[VAL_1:.*]]: !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>):
   !CHECK: {{.*}} = fir.coordinate_of %[[VAL_1:.*]], {{.*}} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.field) -> !fir.ref<i64>
      a = c_loc(b)
   !CHECK: omp.terminator
   !$omp end target data
   !CHECK: }
end subroutine omp_target_device_ptr

 !===============================================================================
 ! Target `use_device_addr` clause
 !===============================================================================

 !CHECK-LABEL: func.func @_QPomp_target_device_addr() {
 subroutine omp_target_device_addr
   integer, pointer :: a
   !CHECK: %[[VAL_0:.*]] = fir.alloca !fir.box<!fir.ptr<i32>> {bindc_name = "a", uniq_name = "_QFomp_target_device_addrEa"}
   !CHECK: %[[MAP_MEMBERS:.*]] = omp.map_info var_ptr({{.*}} : !fir.ref<!fir.box<!fir.ptr<i32>>>, i32) var_ptr_ptr({{.*}} : !fir.llvm_ptr<!fir.ref<i32>>) map_clauses(tofrom) capture(ByRef) -> !fir.llvm_ptr<!fir.ref<i32>> {name = ""}
   !CHECK: %[[MAP:.*]] = omp.map_info var_ptr({{.*}} : !fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.box<!fir.ptr<i32>>) map_clauses(tofrom) capture(ByRef) members(%[[MAP_MEMBERS]] : !fir.llvm_ptr<!fir.ref<i32>>) -> !fir.ref<!fir.box<!fir.ptr<i32>>> {name = "a"}
   !CHECK: omp.target_data map_entries(%[[MAP_MEMBERS]], %[[MAP]] : {{.*}}) use_device_addr(%[[VAL_0]] : !fir.ref<!fir.box<!fir.ptr<i32>>>) {
   !$omp target data map(tofrom: a) use_device_addr(a)
   !CHECK: ^bb0(%[[VAL_1:.*]]: !fir.ref<!fir.box<!fir.ptr<i32>>>):
   !CHECK: {{.*}} = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
      a = 10
   !CHECK: omp.terminator
   !$omp end target data
   !CHECK: }
end subroutine omp_target_device_addr

!===============================================================================
! Target with parallel loop
!===============================================================================

!CHECK-LABEL: func.func @_QPomp_target_parallel_do() {
subroutine omp_target_parallel_do
   !CHECK: %[[C1024:.*]] = arith.constant 1024 : index
   !CHECK: %[[VAL_0:.*]] = fir.alloca !fir.array<1024xi32> {bindc_name = "a", uniq_name = "_QFomp_target_parallel_doEa"}
   integer :: a(1024)
   !CHECK: %[[VAL_1:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFomp_target_parallel_doEi"}
   integer :: i
   !CHECK: %[[C1:.*]] = arith.constant 1 : index
   !CHECK: %[[C0:.*]] = arith.constant 0 : index
   !CHECK: %[[SUB:.*]] = arith.subi %[[C1024]], %[[C1]] : index
   !CHECK: %[[BOUNDS:.*]] = omp.bounds   lower_bound(%[[C0]] : index) upper_bound(%[[SUB]] : index) extent(%[[C1024]] : index) stride(%[[C1]] : index) start_idx(%[[C1]] : index)
   !CHECK: %[[MAP1:.*]] = omp.map_info var_ptr(%[[VAL_0]] : !fir.ref<!fir.array<1024xi32>>, !fir.array<1024xi32>)   map_clauses(tofrom) capture(ByRef) bounds(%[[BOUNDS]]) -> !fir.ref<!fir.array<1024xi32>> {name = "a"}
   !CHECK: %[[MAP2:.*]] = omp.map_info var_ptr(%[[VAL_1]] : !fir.ref<i32>, i32)   map_clauses(implicit, exit_release_or_enter_alloc) capture(ByCopy) -> !fir.ref<i32> {name = "i"}
   !CHECK: omp.target map_entries(%[[MAP1]] -> %[[VAL_2:.*]], %[[MAP2]] -> %[[VAL_3:.*]] : !fir.ref<!fir.array<1024xi32>>, !fir.ref<i32>) {
   !CHECK: ^bb0(%[[VAL_2]]: !fir.ref<!fir.array<1024xi32>>, %[[VAL_3]]: !fir.ref<i32>):
      !CHECK-NEXT: omp.parallel
      !$omp target parallel do map(tofrom: a)
         !CHECK: %[[VAL_4:.*]] = fir.alloca i32 {adapt.valuebyref, pinned}
         !CHECK: %[[VAL_5:.*]] = arith.constant 1 : i32
         !CHECK: %[[VAL_6:.*]] = arith.constant 1024 : i32
         !CHECK: %[[VAL_7:.*]] = arith.constant 1 : i32
         !CHECK: omp.wsloop   for  (%[[VAL_8:.*]]) : i32 = (%[[VAL_5]]) to (%[[VAL_6]]) inclusive step (%[[VAL_7]]) {
         !CHECK: fir.store %[[VAL_8]] to %[[VAL_4]] : !fir.ref<i32>
         !CHECK: %[[VAL_9:.*]] = arith.constant 10 : i32
         !CHECK: %[[VAL_10:.*]] = fir.load %[[VAL_4]] : !fir.ref<i32>
         !CHECK: %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (i32) -> i64
         !CHECK: %[[VAL_12:.*]] = arith.constant 1 : i64
         !CHECK: %[[VAL_13:.*]] = arith.subi %[[VAL_11]], %[[VAL_12]] : i64
         !CHECK: %[[VAL_14:.*]] = fir.coordinate_of %[[VAL_2]], %[[VAL_13]] : (!fir.ref<!fir.array<1024xi32>>, i64) -> !fir.ref<i32>
         !CHECK: fir.store %[[VAL_9]] to %[[VAL_14]] : !fir.ref<i32>
         do i = 1, 1024
            a(i) = 10
         end do
         !CHECK: omp.yield
         !CHECK: }
      !CHECK: omp.terminator
      !CHECK: }
   !CHECK: omp.terminator
   !CHECK: }
   !$omp end target parallel do
 end subroutine omp_target_parallel_do

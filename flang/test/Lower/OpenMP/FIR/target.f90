!RUN: %flang_fc1 -emit-fir -fopenmp %s -o - | FileCheck %s

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
   !CHECK: omp.target   map_entries(%[[MAP]] : !fir.ref<!fir.array<1024xi32>>) {
   !$omp target map(tofrom: a)
      !CHECK: %[[VAL_1:.*]] = arith.constant 10 : i32
      !CHECK: %[[VAL_2:.*]] = arith.constant 1 : i64
      !CHECK: %[[VAL_3:.*]] = arith.constant 1 : i64
      !CHECK: %[[VAL_4:.*]] = arith.subi %[[VAL_2]], %[[VAL_3]] : i64
      !CHECK: %[[VAL_5:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_4]] : (!fir.ref<!fir.array<1024xi32>>, i64) -> !fir.ref<i32>
      !CHECK: fir.store %[[VAL_1]] to %[[VAL_5]] : !fir.ref<i32>
      a(1) = 10
   !CHECK: omp.terminator
   !$omp end target
   !CHECK: }
end subroutine omp_target

!===============================================================================
! Target `thread_limit` clause
!===============================================================================

!CHECK-LABEL: func.func @_QPomp_target_thread_limit() {
subroutine omp_target_thread_limit
   integer :: a
   !CHECK: %[[VAL_1:.*]] = arith.constant 64 : i32
   !CHECK: %[[MAP:.*]] = omp.map_info var_ptr({{.*}})   map_clauses(tofrom) capture(ByRef) -> !fir.ref<i32> {name = "a"}
   !CHECK: omp.target   thread_limit(%[[VAL_1]] : i32) map_entries(%[[MAP]] : !fir.ref<i32>) {
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
   !CHECK: %[[MAP:.*]] = omp.map_info var_ptr({{.*}})   map_clauses(tofrom) capture(ByRef) -> {{.*}} {name = "a"}
   !CHECK: omp.target_data map_entries(%[[MAP]] : {{.*}}) use_device_addr(%[[VAL_0]] : !fir.ref<!fir.box<!fir.ptr<i32>>>) {
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
   !CHECK: %[[MAP:.*]] = omp.map_info var_ptr(%[[VAL_0]] : !fir.ref<!fir.array<1024xi32>>, !fir.array<1024xi32>)   map_clauses(tofrom) capture(ByRef) bounds(%[[BOUNDS]]) -> !fir.ref<!fir.array<1024xi32>> {name = "a"}
   !CHECK: omp.target   map_entries(%[[MAP]] : !fir.ref<!fir.array<1024xi32>>) {
      !CHECK-NEXT: omp.parallel
      !$omp target parallel do map(tofrom: a)
         !CHECK: %[[VAL_2:.*]] = fir.alloca i32 {adapt.valuebyref, pinned}
         !CHECK: %[[VAL_3:.*]] = arith.constant 1 : i32
         !CHECK: %[[VAL_4:.*]] = arith.constant 1024 : i32
         !CHECK: %[[VAL_5:.*]] = arith.constant 1 : i32
         !CHECK: omp.wsloop   for  (%[[VAL_6:.*]]) : i32 = (%[[VAL_3]]) to (%[[VAL_4]]) inclusive step (%[[VAL_5]]) {
         !CHECK: fir.store %[[VAL_6]] to %[[VAL_2]] : !fir.ref<i32>
         !CHECK: %[[VAL_7:.*]] = arith.constant 10 : i32
         !CHECK: %[[VAL_8:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
         !CHECK: %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (i32) -> i64
         !CHECK: %[[VAL_10:.*]] = arith.constant 1 : i64
         !CHECK: %[[VAL_11:.*]] = arith.subi %[[VAL_9]], %[[VAL_10]] : i64
         !CHECK: %[[VAL_12:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_11]] : (!fir.ref<!fir.array<1024xi32>>, i64) -> !fir.ref<i32>
         !CHECK: fir.store %[[VAL_7]] to %[[VAL_12]] : !fir.ref<i32>
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

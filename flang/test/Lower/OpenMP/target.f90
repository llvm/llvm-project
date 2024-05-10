! The "thread_limit" clause was added to the "target" construct in OpenMP 5.1.
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=51 %s -o - | FileCheck %s
!CHECK-LABEL: func.func @_QPomp_target_parallel_do() {
subroutine omp_target_parallel_do
   !CHECK: %[[C1024:.*]] = arith.constant 1024 : index
   !CHECK: %[[VAL_0:.*]] = fir.alloca !fir.array<1024xi32> {bindc_name = "a", uniq_name = "_QFomp_target_parallel_doEa"}
   !CHECK: %[[VAL_0_DECL:.*]]:2 = hlfir.declare %[[VAL_0]](%{{.*}}) {uniq_name = "_QFomp_target_parallel_doEa"} : (!fir.ref<!fir.array<1024xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<1024xi32>>, !fir.ref<!fir.array<1024xi32>>)
   integer :: a(1024)
   !CHECK: %[[VAL_1:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFomp_target_parallel_doEi"}
   !CHECK: %[[VAL_1_DECL:.*]]:2 = hlfir.declare %[[VAL_1]] {uniq_name = "_QFomp_target_parallel_doEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
   integer :: i
   !CHECK: %[[C1:.*]] = arith.constant 1 : index
   !CHECK: %[[C0:.*]] = arith.constant 0 : index
   !CHECK: %[[SUB:.*]] = arith.subi %[[C1024]], %[[C1]] : index
   !CHECK: %[[BOUNDS:.*]] = omp.map.bounds   lower_bound(%[[C0]] : index) upper_bound(%[[SUB]] : index) extent(%[[C1024]] : index) stride(%[[C1]] : index) start_idx(%[[C1]] : index)
   !CHECK: %[[MAP:.*]] = omp.map.info var_ptr(%[[VAL_0_DECL]]#0 : !fir.ref<!fir.array<1024xi32>>, !fir.array<1024xi32>)   map_clauses(tofrom) capture(ByRef) bounds(%[[BOUNDS]]) -> !fir.ref<!fir.array<1024xi32>> {name = "a"}
   !CHECK: omp.target   map_entries(%[[MAP]] -> %[[ARG_0:.*]], %{{.*}} -> %{{.*}} : !fir.ref<!fir.array<1024xi32>>, !fir.ref<i32>) {
   !CHECK: ^bb0(%[[ARG_0]]: !fir.ref<!fir.array<1024xi32>>, %{{.*}}: !fir.ref<i32>):
      !CHECK: %[[VAL_0_DECL:.*]]:2 = hlfir.declare %[[ARG_0]](%{{.*}}) {uniq_name = "_QFomp_target_parallel_doEa"} : (!fir.ref<!fir.array<1024xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<1024xi32>>, !fir.ref<!fir.array<1024xi32>>)
      !CHECK: omp.parallel
      !$omp target parallel do map(tofrom: a)
         !CHECK: %[[I_PVT_ALLOCA:.*]] = fir.alloca i32 {adapt.valuebyref, pinned}
         !CHECK: %[[I_PVT_DECL:.*]]:2 = hlfir.declare %[[I_PVT_ALLOCA]] {uniq_name = "_QFomp_target_parallel_doEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
         !CHECK: omp.wsloop {
         !CHECK-NEXT: omp.loop_nest (%[[I_VAL:.*]]) : i32
         do i = 1, 1024
           !CHECK:   fir.store %[[I_VAL]] to %[[I_PVT_DECL]]#1 : !fir.ref<i32>
           !CHECK:   %[[C10:.*]] = arith.constant 10 : i32
           !CHECK:   %[[I_PVT_VAL:.*]] = fir.load %[[I_PVT_DECL]]#0 : !fir.ref<i32>
           !CHECK:   %[[I_VAL:.*]] = fir.convert %[[I_PVT_VAL]] : (i32) -> i64
           !CHECK:   %[[A_I:.*]] = hlfir.designate %[[VAL_0_DECL]]#0 (%[[I_VAL]])  : (!fir.ref<!fir.array<1024xi32>>, i64) -> !fir.ref<i32>
           !CHECK:   hlfir.assign %[[C10]] to %[[A_I]] : i32, !fir.ref<i32>
            a(i) = 10
         end do
         !CHECK: omp.yield
         !CHECK: }
         !CHECK: omp.terminator
         !CHECK: }
      !CHECK: omp.terminator
      !CHECK: }
   !CHECK: omp.terminator
   !CHECK: }
   !$omp end target parallel do
end subroutine omp_target_parallel_do

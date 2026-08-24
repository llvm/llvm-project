!RUN: bbc -emit-hlfir -fopenmp -fopenmp-version=52 -o - %s 2>&1 | FileCheck %s
!RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %s 2>&1 | FileCheck %s

! Tests lowering of the `iterator` modifier on the `map` clause, for a
! derived-type member accessed through a `declare mapper` and for a plain
! (non-member) array object.

!===============================================================================
! declare mapper (derived-type member: v%a(i))
!===============================================================================

! omp.declare_mapper is always emitted at module scope ahead of any
! func.func, regardless of where its subroutine appears in the source.
! CHECK-LABEL: omp.declare_mapper @_QQFfm :
! CHECK-SAME:    [[TY:!fir\.type<_QFfTs\{a:!fir\.array<10xi32>\}>]] {
! CHECK:       ^bb0(%[[ARG0:.*]]: !fir.ref<[[TY]]>):
! CHECK:         %[[V:.*]]:2 = hlfir.declare %[[ARG0]] {uniq_name = "_QFfEv"} : (!fir.ref<[[TY]]>) -> (!fir.ref<[[TY]]>, !fir.ref<[[TY]]>)
! CHECK:         %[[FIELD:.*]] = fir.coordinate_of %[[V]]#0, a : (!fir.ref<[[TY]]>) -> !fir.ref<!fir.array<10xi32>>
! CHECK:         %[[IT:.*]] = omp.iterator(%[[IV:.*]]: index) = (%{{.*}} to %{{.*}} step %{{.*}}) {
! CHECK:           %[[COOR:.*]] = fir.array_coor %[[FIELD]](%{{.*}}) %{{.*}} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, i64) -> !fir.ref<i32>
! CHECK:           %[[MAPINFO:.*]] = omp.map.info var_ptr(%[[COOR]] : !fir.ref<i32>, i32) map_clauses(tofrom) capture(ByRef) name("a") -> !fir.ref<i32>
! CHECK:           omp.yield(%[[MAPINFO]] : !fir.ref<i32>)
! CHECK:         } -> !omp.iterated<!fir.ref<i32>>
! CHECK:         %[[PARENT:.*]] = omp.map.info var_ptr(%[[V]]#1 : !fir.ref<[[TY]]>, [[TY]]) map_clauses(storage) capture(ByRef) members( :  : ) name("v") partial_map(true) -> !fir.ref<[[TY]]>
! CHECK:         omp.declare_mapper.info map_entries(%[[PARENT]] : !fir.ref<[[TY]]>) map_iterated(%[[IT]] : !omp.iterated<!fir.ref<i32>>)
subroutine f(arg)
  type :: s
    integer :: a(10)
  end type
  type(s) :: arg(:)

  !$omp declare mapper(m: s :: v) map(mapper(m), iterator(i = 1:10): v%a(i))
end

!===============================================================================
! target map (plain, non-member array object)
!===============================================================================

! CHECK-LABEL: func.func @_QPf00(
! CHECK-SAME:    %[[A:.*]]: !fir.ref<!fir.array<10xi32>>
subroutine f00(a)
  integer :: a(10)
  !$omp target map(iterator(i = 1:2): a(i))
  a(1) = 1
  !$omp end target
end
! CHECK: %[[A_DECL:.*]]:2 = hlfir.declare %[[A]](%{{.*}}) dummy_scope %{{.*}} arg 1 {uniq_name = "_QFf00Ea"}
! CHECK: %[[IT:.*]] = omp.iterator(%[[IV:.*]]: index) = (%{{.*}} to %{{.*}} step %{{.*}}) {
! CHECK:   %[[COOR:.*]] = fir.array_coor %[[A_DECL]]#0(%{{.*}}) %{{.*}} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, i64) -> !fir.ref<i32>
! CHECK:   %[[MAPINFO:.*]] = omp.map.info var_ptr(%[[COOR]] : !fir.ref<i32>, i32) map_clauses(tofrom) capture(ByRef) name("a") -> !fir.ref<i32>
! CHECK:   omp.yield(%[[MAPINFO]] : !fir.ref<i32>)
! CHECK: } -> !omp.iterated<!fir.ref<i32>>
! CHECK: omp.target {{.*}}map_iterated(%[[IT]] : !omp.iterated<!fir.ref<i32>>)

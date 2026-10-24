!RUN: bbc -emit-hlfir -fopenmp -fopenmp-version=52 -o - %s 2>&1 | FileCheck %s
!RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %s 2>&1 | FileCheck %s

! Tests that each locator in a `map` clause with an `iterator` modifier is
! expanded using only the ranges its source syntax references, rather than
! every range declared on the clause.
!
! omp.declare_mapper ops are emitted at module scope in reverse declaration
! order, so `f2`'s mapper (declared second) is checked before `f`'s.

!===============================================================================
! An iterator not referenced by a locator (`j` here) must not affect it: only
! `i`'s range is used to build the omp.iterator, which therefore has a single
! induction variable instead of two.
!===============================================================================

! CHECK-LABEL: omp.declare_mapper @_QQFf2m2 :
! CHECK-SAME:    [[TY2:!fir\.type<_QFf2Ts\{a:!fir\.array<10xi32>\}>]] {
! CHECK:       ^bb0(%[[ARG0:.*]]: !fir.ref<[[TY2]]>):
! CHECK:         %[[V2:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK:         %[[FIELD2:.*]] = fir.coordinate_of %[[V2]]#0, a : (!fir.ref<[[TY2]]>) -> !fir.ref<!fir.array<10xi32>>
! CHECK:         %[[IT2:.*]] = omp.iterator(%[[IV:.*]]: i32) = (%c1{{.*}} to %c2{{.*}} step %{{.*}}) {
! CHECK:           %[[COOR2:.*]] = fir.array_coor %[[FIELD2]]
! CHECK:           %[[MAPINFO2:.*]] = omp.map.info var_ptr(%[[COOR2]] : !fir.ref<i32>, i32) map_clauses(tofrom) capture(ByRef) name("a") -> !fir.ref<i32>
! CHECK:           omp.yield(%[[MAPINFO2]] : !fir.ref<i32>)
! CHECK:         } inclusive -> !omp.iterated<!fir.ref<i32>>
! CHECK:         omp.declare_mapper.info map_iterated(%[[IT2]] : !omp.iterated<!fir.ref<i32>>)

!===============================================================================
! An iterator reference removed by constant folding (`1+0*i`) must still make
! this locator iterator-dependent, so the empty `i` range (3:2) suppresses it
! at run time instead of it being folded into a plain, always-mapped `v%a(1)`.
! This also confirms no parent "storage" map is synthesized for a mapper
! whose only locator is iterator-driven (see map-iterator.f90).
!===============================================================================

! CHECK-LABEL: omp.declare_mapper @_QQFfm :
! CHECK-SAME:    [[TY:!fir\.type<_QFfTs\{a:!fir\.array<10xi32>\}>]] {
! CHECK:       ^bb0(%[[ARG0:.*]]: !fir.ref<[[TY]]>):
! CHECK:         %[[V:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK:         %[[FIELD:.*]] = fir.coordinate_of %[[V]]#0, a : (!fir.ref<[[TY]]>) -> !fir.ref<!fir.array<10xi32>>
! CHECK:         %[[IT:.*]] = omp.iterator(%{{.*}}: i32) = (%c3{{.*}} to %c2{{.*}} step %{{.*}}) {
! CHECK:           %[[COOR:.*]] = fir.array_coor %[[FIELD]]{{.*}} %c1{{.*}} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, i64) -> !fir.ref<i32>
! CHECK:           %[[MAPINFO:.*]] = omp.map.info var_ptr(%[[COOR]] : !fir.ref<i32>, i32) map_clauses(tofrom) capture(ByRef) name("a") -> !fir.ref<i32>
! CHECK:           omp.yield(%[[MAPINFO]] : !fir.ref<i32>)
! CHECK:         } inclusive -> !omp.iterated<!fir.ref<i32>>
! CHECK:         omp.declare_mapper.info map_iterated(%[[IT]] : !omp.iterated<!fir.ref<i32>>)
subroutine f(arg)
  type :: s
    integer :: a(10)
  end type
  type(s) :: arg(:)

  !$omp declare mapper(m: s :: v) map(mapper(m), iterator(i = 3:2): v%a(1+0*i))
end

subroutine f2(arg)
  type :: s
    integer :: a(10)
  end type
  type(s) :: arg(:)

  !$omp declare mapper(m2: s :: v) map(mapper(m2), iterator(i = 1:2, j = 3:2): v%a(i))
end

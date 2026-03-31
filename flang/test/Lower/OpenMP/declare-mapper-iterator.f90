! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %s | FileCheck %s

subroutine declare_mapper_iterator(arg)
  type :: s
    integer :: a(10)
  end type
  type(s) :: arg(:)

  !$omp declare mapper(m: s :: v) map(iterator(i = 1:10): v%a(i))
end

! CHECK-LABEL: omp.declare_mapper
! CHECK: ^bb0(%[[ARG:.*]]: !fir.ref<!fir.type<_QFdeclare_mapper_iteratorTs{{.*}}>):
! CHECK:   %[[DECL:.*]]:2 = hlfir.declare %[[ARG]] {uniq_name = "_QFdeclare_mapper_iteratorEv"}
! CHECK:   %[[IT:.*]] = omp.iterator(%[[IV:.*]]: index) = ({{.*}} to {{.*}} step {{.*}}) {
! CHECK:     %[[IV_I32:.*]] = fir.convert %[[IV]] : (index) -> i32
! CHECK:     %[[A:.*]] = hlfir.designate %[[DECL]]#0{"a"} {{.*}} : (!fir.ref<!fir.type<_QFdeclare_mapper_iteratorTs{{.*}}>>, !fir.shape<1>) -> !fir.ref<!fir.array<10xi32>>
! CHECK:     %[[IV_I64:.*]] = fir.convert %[[IV_I32]] : (i32) -> i64
! CHECK:     %[[IV_IDX:.*]] = fir.convert %[[IV_I64]] : (i64) -> index
! CHECK:     %[[LB:.*]] = arith.subi %[[IV_IDX]], %{{.*}} : index
! CHECK:     %[[BOUNDS:.*]] = omp.map.bounds lower_bound(%[[LB]] : index) upper_bound(%[[LB]] : index) extent(%{{.*}} : index) stride(%{{.*}} : index) start_idx(%{{.*}} : index)
! CHECK:     %[[MAP:.*]] = omp.map.info var_ptr(%[[A]] : !fir.ref<!fir.array<10xi32>>, !fir.array<10xi32>) map_clauses(tofrom) capture(ByRef) bounds(%[[BOUNDS]]) -> !llvm.ptr {name = ""}
! CHECK:     omp.yield(%[[MAP]] : !llvm.ptr)
! CHECK:   } -> !omp.iterated<!llvm.ptr>
! CHECK:   omp.declare_mapper.info map_iterated(%[[IT]] : !omp.iterated<!llvm.ptr>)

! RUN: %flang_fc1 -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s

subroutine TestCharLenBounds(clen)

  character(len=*) :: clen

  !$omp target map(clen)
  !$omp end target
end subroutine TestCharLenBounds

!CHECK: %[[DUMMY:.*]] = fir.dummy_scope : !fir.dscope
!CHECK: %[[UNBOX:.*]]:2 = fir.unboxchar %{{.*}} : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
!CHECK: %[[DECLARE:.*]]:2 = hlfir.declare %[[UNBOX]]#0 typeparams %2#1 dummy_scope %[[DUMMY]] arg {{[0-9]+}} {uniq_name = "_QFtestcharlenboundsEclen"} : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
!CHECK: %[[LB_START_IDX:.*]] = arith.constant 0 : index
!CHECK: %[[STRIDE:.*]] = arith.constant 1 : index
!CHECK: %[[EXTENT:.*]]:2 = fir.unboxchar %[[DECLARE]]#0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
!CHECK: %[[UB:.*]] = arith.subi %[[EXTENT]]#1, %[[STRIDE]] : index
!CHECK: %[[BOUNDS:.*]] = omp.map.bounds lower_bound(%[[LB_START_IDX]] : index) upper_bound(%[[UB]] : index) extent(%[[EXTENT]]#1 : index) stride(%[[STRIDE]] : index) start_idx(%[[LB_START_IDX]] : index) {stride_in_bytes = true}
!CHECK: %{{.*}} = omp.map.info {{.*}} bounds(%[[BOUNDS]]) {{.*}}

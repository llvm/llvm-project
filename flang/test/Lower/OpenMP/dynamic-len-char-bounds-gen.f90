! RUN: %flang_fc1 -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s

subroutine TestCharLenBounds(clen)

  character(len=*) :: clen

  !$omp target map(clen)
  !$omp end target
end subroutine TestCharLenBounds

!CHECK: %[[DUMMY:.*]] = fir.dummy_scope : !fir.dscope
!CHECK: %[[UNBOX:.*]]:2 = fir.unboxchar %{{.*}} : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
!CHECK: %[[DECLARE:.*]]:2 = hlfir.declare %[[UNBOX]]#0 typeparams %2#1 dummy_scope %[[DUMMY]] arg {{[0-9]+}} uniq_name("_QFtestcharlenboundsEclen") : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
!CHECK: %[[LB_START_IDX:.*]] = arith.constant 0 : index
!CHECK: %[[STRIDE:.*]] = arith.constant 1 : index
!CHECK: %[[EXTENT:.*]]:2 = fir.unboxchar %[[DECLARE]]#0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
!CHECK: %[[UB:.*]] = arith.subi %[[EXTENT]]#1, %[[STRIDE]] : index
!CHECK: %[[BOUNDS:.*]] = omp.map.bounds lower_bound(%[[LB_START_IDX]] : index) upper_bound(%[[UB]] : index) extent(%[[EXTENT]]#1 : index) stride(%[[STRIDE]] : index) start_idx(%[[LB_START_IDX]] : index) stride_in_bytes(true)
!CHECK: %{{.*}} = omp.map.info {{.*}} bounds(%[[BOUNDS]]) {{.*}}

! A dynamic length is a count of code units. Wide-character byte strides
! must include the kind's storage size, in both motion directions.
! CHECK-LABEL: func.func @_QPupdate_wide_char(
! CHECK: %[[ZERO:.*]] = arith.constant 0 : index
! CHECK: %[[ONE:.*]] = arith.constant 1 : index
! CHECK: %[[WIDTH:.*]] = arith.constant 4 : index
! CHECK: %[[UNBOX:.*]]:2 = fir.unboxchar {{.*}} : (!fir.boxchar<4>) -> (!fir.ref<!fir.char<4,?>>, index)
! CHECK: %[[UB:.*]] = arith.subi %[[UNBOX]]#1, %[[ONE]] : index
! CHECK: %[[BOUNDS:.*]] = omp.map.bounds lower_bound(%[[ZERO]] : index) upper_bound(%[[UB]] : index) extent(%[[UNBOX]]#1 : index) stride(%[[WIDTH]] : index) start_idx(%[[ZERO]] : index) stride_in_bytes(true)
! CHECK: omp.map.info {{.*}} map_clauses(to) {{.*}} bounds(%[[BOUNDS]])
! CHECK: omp.target_update
! CHECK: %[[WIDTH_FROM:.*]] = arith.constant 4 : index
! CHECK: omp.map.bounds {{.*}} stride(%[[WIDTH_FROM]] : index) {{.*}} stride_in_bytes(true)
! CHECK: omp.map.info {{.*}} map_clauses(from)
! CHECK: omp.target_update
subroutine update_wide_char(clen)
  character(kind=4,len=*) :: clen
  !$omp target update to(clen)
  !$omp target update from(clen)
end subroutine

! CHECK-LABEL: func.func @_QPupdate_optional_wide_char(
! CHECK: %[[WIDTH:.*]] = arith.constant 2 : index
! CHECK: %[[PRESENT_BOUNDS:.*]]:2 = fir.if {{.*}} -> (index, index) {
! CHECK: fir.result {{.*}}, %[[WIDTH]] : index, index
! CHECK: } else {
! CHECK: fir.result %{{.*}}, %{{.*}} : index, index
! CHECK: omp.map.bounds {{.*}} stride(%[[PRESENT_BOUNDS]]#1 : index) {{.*}} stride_in_bytes(true)
! CHECK: omp.target_update
subroutine update_optional_wide_char(clen)
  character(kind=2,len=*), optional :: clen
  !$omp target update to(clen)
end subroutine

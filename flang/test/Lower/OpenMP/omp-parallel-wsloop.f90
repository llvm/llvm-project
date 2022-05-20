! This test checks lowering of OpenMP DO Directive (Worksharing).

! RUN: bbc -fopenmp -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPsimple_parallel_do()
subroutine simple_parallel_do
  integer :: i
  ! CHECK:  omp.parallel
  ! CHECK:     %[[WS_LB:.*]] = arith.constant 1 : i32
  ! CHECK:     %[[WS_UB:.*]] = arith.constant 9 : i32
  ! CHECK:     %[[WS_STEP:.*]] = arith.constant 1 : i32
  ! CHECK:     omp.wsloop for (%[[I:.*]]) : i32 = (%[[WS_LB]]) to (%[[WS_UB]]) inclusive step (%[[WS_STEP]])
  !$OMP PARALLEL DO
  do i=1, 9
  ! CHECK:    fir.call @_FortranAioOutputInteger32({{.*}}, %[[I]]) : (!fir.ref<i8>, i32) -> i1
    print*, i
  end do
  ! CHECK:       omp.yield
  ! CHECK:       omp.terminator
  !$OMP END PARALLEL DO
end subroutine

! CHECK-LABEL: func @_QPparallel_do_with_parallel_clauses
! CHECK-SAME: %[[COND_REF:.*]]: !fir.ref<!fir.logical<4>> {fir.bindc_name = "cond"}, %[[NT_REF:.*]]: !fir.ref<i32> {fir.bindc_name = "nt"}
subroutine parallel_do_with_parallel_clauses(cond, nt)
  logical :: cond
  integer :: nt
  integer :: i
  ! CHECK:  %[[COND:.*]] = fir.load %[[COND_REF]] : !fir.ref<!fir.logical<4>>
  ! CHECK:  %[[COND_CVT:.*]] = fir.convert %[[COND]] : (!fir.logical<4>) -> i1
  ! CHECK:  %[[NT:.*]] = fir.load %[[NT_REF]] : !fir.ref<i32>
  ! CHECK:  omp.parallel if(%[[COND_CVT]] : i1) num_threads(%[[NT]] : i32) proc_bind(close)
  ! CHECK:     %[[WS_LB:.*]] = arith.constant 1 : i32
  ! CHECK:     %[[WS_UB:.*]] = arith.constant 9 : i32
  ! CHECK:     %[[WS_STEP:.*]] = arith.constant 1 : i32
  ! CHECK:     omp.wsloop for (%[[I:.*]]) : i32 = (%[[WS_LB]]) to (%[[WS_UB]]) inclusive step (%[[WS_STEP]])
  !$OMP PARALLEL DO IF(cond) NUM_THREADS(nt) PROC_BIND(close)
  do i=1, 9
  ! CHECK:    fir.call @_FortranAioOutputInteger32({{.*}}, %[[I]]) : (!fir.ref<i8>, i32) -> i1
    print*, i
  end do
  ! CHECK:       omp.yield
  ! CHECK:       omp.terminator
  !$OMP END PARALLEL DO
end subroutine

! CHECK-LABEL: func @_QPparallel_do_with_clauses
! CHECK-SAME: %[[NT_REF:.*]]: !fir.ref<i32> {fir.bindc_name = "nt"}
subroutine parallel_do_with_clauses(nt)
  integer :: nt
  integer :: i
  ! CHECK:  %[[NT:.*]] = fir.load %[[NT_REF]] : !fir.ref<i32>
  ! CHECK:  omp.parallel num_threads(%[[NT]] : i32)
  ! CHECK:     %[[WS_LB:.*]] = arith.constant 1 : i32
  ! CHECK:     %[[WS_UB:.*]] = arith.constant 9 : i32
  ! CHECK:     %[[WS_STEP:.*]] = arith.constant 1 : i32
  ! CHECK:     omp.wsloop schedule(dynamic) for (%[[I:.*]]) : i32 = (%[[WS_LB]]) to (%[[WS_UB]]) inclusive step (%[[WS_STEP]])
  !$OMP PARALLEL DO NUM_THREADS(nt) SCHEDULE(dynamic)
  do i=1, 9
  ! CHECK:    fir.call @_FortranAioOutputInteger32({{.*}}, %[[I]]) : (!fir.ref<i8>, i32) -> i1
    print*, i
  end do
  ! CHECK:       omp.yield
  ! CHECK:       omp.terminator
  !$OMP END PARALLEL DO
end subroutine

! CHECK-LABEL: func @_QPparallel_do_with_privatisation_clauses
! CHECK-SAME: %[[COND_REF:.*]]: !fir.ref<!fir.logical<4>> {fir.bindc_name = "cond"}, %[[NT_REF:.*]]: !fir.ref<i32> {fir.bindc_name = "nt"}
subroutine parallel_do_with_privatisation_clauses(cond,nt)
  logical :: cond
  integer :: nt
  integer :: i
  ! CHECK:  omp.parallel
  ! CHECK:    %[[PRIVATE_COND_REF:.*]] = fir.alloca !fir.logical<4> {bindc_name = "cond", pinned, uniq_name = "_QFparallel_do_with_privatisation_clausesEcond"}
  ! CHECK:    %[[PRIVATE_NT_REF:.*]] = fir.alloca i32 {bindc_name = "nt", pinned, uniq_name = "_QFparallel_do_with_privatisation_clausesEnt"}
  ! CHECK:    %[[NT_VAL:.*]] = fir.load %[[NT_REF]] : !fir.ref<i32>
  ! CHECK:    fir.store %[[NT_VAL]] to %[[PRIVATE_NT_REF]] : !fir.ref<i32>
  ! CHECK:    %[[WS_LB:.*]] = arith.constant 1 : i32
  ! CHECK:    %[[WS_UB:.*]] = arith.constant 9 : i32
  ! CHECK:    %[[WS_STEP:.*]] = arith.constant 1 : i32
  ! CHECK:    omp.wsloop for (%[[I:.*]]) : i32 = (%[[WS_LB]]) to (%[[WS_UB]]) inclusive step (%[[WS_STEP]])
  !$OMP PARALLEL DO PRIVATE(cond) FIRSTPRIVATE(nt)
  do i=1, 9
  ! CHECK:      fir.call @_FortranAioOutputInteger32({{.*}}, %[[I]]) : (!fir.ref<i8>, i32) -> i1
  ! CHECK:      %[[PRIVATE_COND_VAL:.*]] = fir.load %[[PRIVATE_COND_REF]] : !fir.ref<!fir.logical<4>>
  ! CHECK:      %[[PRIVATE_COND_VAL_CVT:.*]] = fir.convert %[[PRIVATE_COND_VAL]] : (!fir.logical<4>) -> i1
  ! CHECK:      fir.call @_FortranAioOutputLogical({{.*}}, %[[PRIVATE_COND_VAL_CVT]]) : (!fir.ref<i8>, i1) -> i1
  ! CHECK:      %[[PRIVATE_NT_VAL:.*]] = fir.load %[[PRIVATE_NT_REF]] : !fir.ref<i32>
  ! CHECK:      fir.call @_FortranAioOutputInteger32({{.*}}, %[[PRIVATE_NT_VAL]]) : (!fir.ref<i8>, i32) -> i1
    print*, i, cond, nt
  end do
  ! CHECK:      omp.yield
  ! CHECK:    omp.terminator
  !$OMP END PARALLEL DO
end subroutine

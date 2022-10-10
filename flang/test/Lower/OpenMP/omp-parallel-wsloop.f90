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
  ! CHECK:    fir.store %[[I]] to %[[IV_ADDR:.*]] : !fir.ref<i32>
  ! CHECK:    %[[LOAD_IV:.*]] = fir.load %[[IV_ADDR]] : !fir.ref<i32>
  ! CHECK:    fir.call @_FortranAioOutputInteger32({{.*}}, %[[LOAD_IV]]) : (!fir.ref<i8>, i32) -> i1
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
  ! CHECK:    fir.store %[[I]] to %[[IV_ADDR:.*]] : !fir.ref<i32>
  ! CHECK:    %[[LOAD_IV:.*]] = fir.load %[[IV_ADDR]] : !fir.ref<i32>
  ! CHECK:    fir.call @_FortranAioOutputInteger32({{.*}}, %[[LOAD_IV]]) : (!fir.ref<i8>, i32) -> i1
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
  ! CHECK:    fir.store %[[I]] to %[[IV_ADDR:.*]] : !fir.ref<i32>
  ! CHECK:    %[[LOAD_IV:.*]] = fir.load %[[IV_ADDR]] : !fir.ref<i32>
  ! CHECK:    fir.call @_FortranAioOutputInteger32({{.*}}, %[[LOAD_IV]]) : (!fir.ref<i8>, i32) -> i1
    print*, i
  end do
  ! CHECK:       omp.yield
  ! CHECK:       omp.terminator
  !$OMP END PARALLEL DO
end subroutine

!===============================================================================
! Checking for the following construct:
!   !$omp parallel do private(...) firstprivate(...)
!===============================================================================

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
  ! CHECK:    fir.store %[[I]] to %[[IV_ADDR:.*]] : !fir.ref<i32>
  ! CHECK:    %[[LOAD_IV:.*]] = fir.load %[[IV_ADDR]] : !fir.ref<i32>
  ! CHECK:      fir.call @_FortranAioOutputInteger32({{.*}}, %[[LOAD_IV]]) : (!fir.ref<i8>, i32) -> i1
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

!===============================================================================
! Checking for the following construct
!   !$omp parallel private(...) firstprivate(...)
!   !$omp do
!===============================================================================

subroutine parallel_private_do(cond,nt)
logical :: cond
  integer :: nt
  integer :: i
  !$OMP PARALLEL PRIVATE(cond) FIRSTPRIVATE(nt)
  !$OMP DO
  do i=1, 9
    call foo(i, cond, nt)
  end do
  !$OMP END DO
  !$OMP END PARALLEL
end subroutine parallel_private_do

! CHECK-LABEL:   func.func @_QPparallel_private_do(
! CHECK-SAME:                                      %[[VAL_0:.*]]: !fir.ref<!fir.logical<4>> {fir.bindc_name = "cond"},
! CHECK-SAME:                                      %[[VAL_1:.*]]: !fir.ref<i32> {fir.bindc_name = "nt"}) {
! CHECK:           %[[I:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFparallel_private_doEi"}
! CHECK:           omp.parallel   {
! CHECK:             %[[I_PRIV:.*]] = fir.alloca i32 {adapt.valuebyref, pinned}
! CHECK:             %[[COND_ADDR:.*]] = fir.alloca !fir.logical<4> {bindc_name = "cond", pinned, uniq_name = "_QFparallel_private_doEcond"}
! CHECK:             %[[NT_ADDR:.*]] = fir.alloca i32 {bindc_name = "nt", pinned, uniq_name = "_QFparallel_private_doEnt"}
! CHECK:             %[[NT:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
! CHECK:             fir.store %[[NT]] to %[[NT_ADDR]] : !fir.ref<i32>
! CHECK:             %[[VAL_7:.*]] = arith.constant 1 : i32
! CHECK:             %[[VAL_8:.*]] = arith.constant 9 : i32
! CHECK:             %[[VAL_9:.*]] = arith.constant 1 : i32
! CHECK:             omp.wsloop   for  (%[[I:.*]]) : i32 = (%[[VAL_7]]) to (%[[VAL_8]]) inclusive step (%[[VAL_9]]) {
! CHECK:               fir.store %[[I]] to %[[I_PRIV]] : !fir.ref<i32>
! CHECK:               fir.call @_QPfoo(%[[I_PRIV]], %[[COND_ADDR]], %[[NT_ADDR]]) : (!fir.ref<i32>, !fir.ref<!fir.logical<4>>, !fir.ref<i32>) -> ()
! CHECK:               omp.yield
! CHECK:             }
! CHECK:             omp.terminator
! CHECK:           }
! CHECK:           return
! CHECK:         }

!===============================================================================
! Checking for the following construct
!   !$omp parallel
!   !$omp do firstprivate(...) firstprivate(...)
!===============================================================================

subroutine omp_parallel_multiple_firstprivate_do(a, b)
  integer::a, b
  !$OMP PARALLEL FIRSTPRIVATE(a) FIRSTPRIVATE(b)
  !$OMP DO
  do i=1, 10
    call bar(i, a)
  end do
  !$OMP END DO
  !$OMP END PARALLEL
end subroutine omp_parallel_multiple_firstprivate_do

! CHECK-LABEL:   func.func @_QPomp_parallel_multiple_firstprivate_do(
! CHECK-SAME:                                                        %[[A_ADDR:.*]]: !fir.ref<i32> {fir.bindc_name = "a"},
! CHECK-SAME:                                                        %[[B_ADDR:.*]]: !fir.ref<i32> {fir.bindc_name = "b"}) {
! CHECK:           %[[I_ADDR:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFomp_parallel_multiple_firstprivate_doEi"}
! CHECK:           omp.parallel   {
! CHECK:             %[[I_PRIV_ADDR:.*]] = fir.alloca i32 {adapt.valuebyref, pinned}
! CHECK:             %[[A_PRIV_ADDR:.*]] = fir.alloca i32 {bindc_name = "a", pinned, uniq_name = "_QFomp_parallel_multiple_firstprivate_doEa"}
! CHECK:             %[[A:.*]] = fir.load %[[A_ADDR]] : !fir.ref<i32>
! CHECK:             fir.store %[[A]] to %[[A_PRIV_ADDR]] : !fir.ref<i32>
! CHECK:             %[[B_PRIV_ADDR:.*]] = fir.alloca i32 {bindc_name = "b", pinned, uniq_name = "_QFomp_parallel_multiple_firstprivate_doEb"}
! CHECK:             %[[B:.*]] = fir.load %[[B_ADDR]] : !fir.ref<i32>
! CHECK:             fir.store %[[B]] to %[[B_PRIV_ADDR]] : !fir.ref<i32>
! CHECK:             %[[VAL_8:.*]] = arith.constant 1 : i32
! CHECK:             %[[VAL_9:.*]] = arith.constant 10 : i32
! CHECK:             %[[VAL_10:.*]] = arith.constant 1 : i32
! CHECK:             omp.wsloop   for  (%[[I:.*]]) : i32 = (%[[VAL_8]]) to (%[[VAL_9]]) inclusive step (%[[VAL_10]]) {
! CHECK:               fir.store %[[I]] to %[[I_PRIV_ADDR]] : !fir.ref<i32>
! CHECK:               fir.call @_QPbar(%[[I_PRIV_ADDR]], %[[A_PRIV_ADDR]]) : (!fir.ref<i32>, !fir.ref<i32>) -> ()
! CHECK:               omp.yield
! CHECK:             }
! CHECK:             omp.terminator
! CHECK:           }
! CHECK:           return
! CHECK:         }

!===============================================================================
! Checking for the following construct
!   !$omp parallel
!   !$omp do private(...) firstprivate(...)
!===============================================================================

subroutine parallel_do_private(cond,nt)
logical :: cond
  integer :: nt
  integer :: i
  !$OMP PARALLEL
  !$OMP DO PRIVATE(cond) FIRSTPRIVATE(nt)
  do i=1, 9
    call foo(i, cond, nt)
  end do
  !$OMP END DO
  !$OMP END PARALLEL
end subroutine parallel_do_private

! CHECK-LABEL:   func.func @_QPparallel_do_private(
! CHECK-SAME:                                      %[[VAL_0:.*]]: !fir.ref<!fir.logical<4>> {fir.bindc_name = "cond"},
! CHECK-SAME:                                      %[[VAL_1:.*]]: !fir.ref<i32> {fir.bindc_name = "nt"}) {
! CHECK:           %[[I_ADDR:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFparallel_do_privateEi"}
! CHECK:           omp.parallel   {
! CHECK:             %[[COND_ADDR:.*]] = fir.alloca !fir.logical<4> {bindc_name = "cond", pinned, uniq_name = "_QFparallel_do_privateEcond"}
! CHECK:             %[[NT_ADDR:.*]] = fir.alloca i32 {bindc_name = "nt", pinned, uniq_name = "_QFparallel_do_privateEnt"}
! CHECK:             %[[NT:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
! CHECK:             fir.store %[[NT]] to %[[NT_ADDR]] : !fir.ref<i32>
! CHECK:             %[[I_PRIV_ADDR:.*]] = fir.alloca i32 {adapt.valuebyref, pinned}
! CHECK:             %[[VAL_7:.*]] = arith.constant 1 : i32
! CHECK:             %[[VAL_8:.*]] = arith.constant 9 : i32
! CHECK:             %[[VAL_9:.*]] = arith.constant 1 : i32
! CHECK:             omp.wsloop   for  (%[[I:.*]]) : i32 = (%[[VAL_7]]) to (%[[VAL_8]]) inclusive step (%[[VAL_9]]) {
! CHECK:               fir.store %[[I]] to %[[I_PRIV_ADDR]] : !fir.ref<i32>
! CHECK:               fir.call @_QPfoo(%[[I_PRIV_ADDR]], %[[COND_ADDR]], %[[NT_ADDR]]) : (!fir.ref<i32>, !fir.ref<!fir.logical<4>>, !fir.ref<i32>) -> ()
! CHECK:               omp.yield
! CHECK:             }
! CHECK:             omp.terminator
! CHECK:           }
! CHECK:           return
! CHECK:         }

!===============================================================================
! Checking for the following construct
!   !$omp parallel
!   !$omp do firstprivate(...) firstprivate(...)
!===============================================================================

subroutine omp_parallel_do_multiple_firstprivate(a, b)
  integer::a, b
  !$OMP PARALLEL
  !$OMP DO FIRSTPRIVATE(a) FIRSTPRIVATE(b)
  do i=1, 10
    call bar(i, a)
  end do
  !$OMP END DO
  !$OMP END PARALLEL
end subroutine omp_parallel_do_multiple_firstprivate

! CHECK-LABEL:   func.func @_QPomp_parallel_do_multiple_firstprivate(
! CHECK-SAME:                                                        %[[A_ADDR:.*]]: !fir.ref<i32> {fir.bindc_name = "a"},
! CHECK-SAME:                                                        %[[B_ADDR:.*]]: !fir.ref<i32> {fir.bindc_name = "b"}) {
! CHECK:           %[[I_ADDR:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFomp_parallel_do_multiple_firstprivateEi"}
! CHECK:           omp.parallel   {
! CHECK:             %[[A_PRIV_ADDR:.*]] = fir.alloca i32 {bindc_name = "a", pinned, uniq_name = "_QFomp_parallel_do_multiple_firstprivateEa"}
! CHECK:             %[[A:.*]] = fir.load %[[A_ADDR]] : !fir.ref<i32>
! CHECK:             fir.store %[[A]] to %[[A_PRIV_ADDR]] : !fir.ref<i32>
! CHECK:             %[[B_PRIV_ADDR:.*]] = fir.alloca i32 {bindc_name = "b", pinned, uniq_name = "_QFomp_parallel_do_multiple_firstprivateEb"}
! CHECK:             %[[B:.*]] = fir.load %[[B_ADDR]] : !fir.ref<i32>
! CHECK:             fir.store %[[B]] to %[[B_PRIV_ADDR]] : !fir.ref<i32>
! CHECK:             %[[I_PRIV_ADDR:.*]] = fir.alloca i32 {adapt.valuebyref, pinned}
! CHECK:             %[[VAL_8:.*]] = arith.constant 1 : i32
! CHECK:             %[[VAL_9:.*]] = arith.constant 10 : i32
! CHECK:             %[[VAL_10:.*]] = arith.constant 1 : i32
! CHECK:             omp.wsloop   for  (%[[I:.*]]) : i32 = (%[[VAL_8]]) to (%[[VAL_9]]) inclusive step (%[[VAL_10]]) {
! CHECK:               fir.store %[[I]] to %[[I_PRIV_ADDR]] : !fir.ref<i32>
! CHECK:               fir.call @_QPbar(%[[I_PRIV_ADDR]], %[[A_PRIV_ADDR]]) : (!fir.ref<i32>, !fir.ref<i32>) -> ()
! CHECK:               omp.yield
! CHECK:             }
! CHECK:             omp.terminator
! CHECK:           }
! CHECK:           return
! CHECK:         }

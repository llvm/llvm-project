! This test checks lowering of OpenMP DO Directive (Worksharing).

! RUN: bbc -fopenmp -emit-hlfir %s -o - \
! RUN: | FileCheck %s

! CHECK-LABEL: func @_QPsimple_parallel_do()
subroutine simple_parallel_do
  integer :: i
  ! CHECK:  omp.parallel
  ! CHECK:      %[[WS_LB:.*]] = arith.constant 1 : i32
  ! CHECK:      %[[WS_UB:.*]] = arith.constant 9 : i32
  ! CHECK:      %[[WS_STEP:.*]] = arith.constant 1 : i32
  ! CHECK:      omp.wsloop {
  ! CHECK-NEXT: omp.loop_nest (%[[I:.*]]) : i32 = (%[[WS_LB]]) to (%[[WS_UB]]) inclusive step (%[[WS_STEP]]) {
  !$OMP PARALLEL DO
  do i=1, 9
  ! CHECK:      fir.store %[[I]] to %[[IV_ADDR:.*]]#1 : !fir.ref<i32>
  ! CHECK:      %[[LOAD_IV:.*]] = fir.load %[[IV_ADDR]]#0 : !fir.ref<i32>
  ! CHECK:      fir.call @_FortranAioOutputInteger32({{.*}}, %[[LOAD_IV]]) {{.*}}: (!fir.ref<i8>, i32) -> i1
    print*, i
  end do
  ! CHECK:      omp.yield
  ! CHECK:      omp.terminator
  !$OMP END PARALLEL DO
end subroutine

! CHECK-LABEL: func @_QPparallel_do_with_parallel_clauses
! CHECK-SAME: %[[COND_REF:.*]]: !fir.ref<!fir.logical<4>> {fir.bindc_name = "cond"}, %[[NT_REF:.*]]: !fir.ref<i32> {fir.bindc_name = "nt"}
subroutine parallel_do_with_parallel_clauses(cond, nt)
  ! CHECK: %[[COND_DECL:.*]]:2 = hlfir.declare %[[COND_REF]] dummy_scope %{{[0-9]+}} {uniq_name = "_QFparallel_do_with_parallel_clausesEcond"} : (!fir.ref<!fir.logical<4>>, !fir.dscope) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
  ! CHECK: %[[NT_DECL:.*]]:2 = hlfir.declare %[[NT_REF]] dummy_scope %{{[0-9]+}} {uniq_name = "_QFparallel_do_with_parallel_clausesEnt"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  logical :: cond
  integer :: nt
  integer :: i
  ! CHECK:  %[[COND:.*]] = fir.load %[[COND_DECL]]#0 : !fir.ref<!fir.logical<4>>
  ! CHECK:  %[[COND_CVT:.*]] = fir.convert %[[COND]] : (!fir.logical<4>) -> i1
  ! CHECK:  %[[NT:.*]] = fir.load %[[NT_DECL]]#0 : !fir.ref<i32>
  ! CHECK:  omp.parallel if(%[[COND_CVT]]) num_threads(%[[NT]] : i32) proc_bind(close)
  ! CHECK:      %[[WS_LB:.*]] = arith.constant 1 : i32
  ! CHECK:      %[[WS_UB:.*]] = arith.constant 9 : i32
  ! CHECK:      %[[WS_STEP:.*]] = arith.constant 1 : i32
  ! CHECK:      omp.wsloop {
  ! CHECK-NEXT: omp.loop_nest (%[[I:.*]]) : i32 = (%[[WS_LB]]) to (%[[WS_UB]]) inclusive step (%[[WS_STEP]]) {
  !$OMP PARALLEL DO IF(cond) NUM_THREADS(nt) PROC_BIND(close)
  do i=1, 9
  ! CHECK:      fir.store %[[I]] to %[[IV_ADDR:.*]]#1 : !fir.ref<i32>
  ! CHECK:      %[[LOAD_IV:.*]] = fir.load %[[IV_ADDR]]#0 : !fir.ref<i32>
  ! CHECK:      fir.call @_FortranAioOutputInteger32({{.*}}, %[[LOAD_IV]]) {{.*}}: (!fir.ref<i8>, i32) -> i1
    print*, i
  end do
  ! CHECK:      omp.yield
  ! CHECK:      omp.terminator
  !$OMP END PARALLEL DO
end subroutine

! CHECK-LABEL: func @_QPparallel_do_with_clauses
! CHECK-SAME: %[[NT_REF:.*]]: !fir.ref<i32> {fir.bindc_name = "nt"}
subroutine parallel_do_with_clauses(nt)
  ! CHECK:  %[[NT_DECL:.*]]:2 = hlfir.declare %[[NT_REF]] dummy_scope %{{[0-9]+}} {uniq_name = "_QFparallel_do_with_clausesEnt"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  integer :: nt
  integer :: i
  ! CHECK:  %[[NT:.*]] = fir.load %[[NT_DECL]]#0 : !fir.ref<i32>
  ! CHECK:  omp.parallel num_threads(%[[NT]] : i32)
  ! CHECK:      %[[WS_LB:.*]] = arith.constant 1 : i32
  ! CHECK:      %[[WS_UB:.*]] = arith.constant 9 : i32
  ! CHECK:      %[[WS_STEP:.*]] = arith.constant 1 : i32
  ! CHECK:      omp.wsloop schedule(dynamic) {
  ! CHECK-NEXT: omp.loop_nest (%[[I:.*]]) : i32 = (%[[WS_LB]]) to (%[[WS_UB]]) inclusive step (%[[WS_STEP]]) {
  !$OMP PARALLEL DO NUM_THREADS(nt) SCHEDULE(dynamic)
  do i=1, 9
  ! CHECK:      fir.store %[[I]] to %[[IV_ADDR:.*]]#1 : !fir.ref<i32>
  ! CHECK:      %[[LOAD_IV:.*]] = fir.load %[[IV_ADDR]]#0 : !fir.ref<i32>
  ! CHECK:      fir.call @_FortranAioOutputInteger32({{.*}}, %[[LOAD_IV]]) {{.*}}: (!fir.ref<i8>, i32) -> i1
    print*, i
  end do
  ! CHECK:      omp.yield
  ! CHECK:      omp.terminator
  !$OMP END PARALLEL DO
end subroutine

!===============================================================================
! Checking for the following construct:
!   !$omp parallel do private(...) firstprivate(...)
!===============================================================================

! CHECK-LABEL: func @_QPparallel_do_with_privatisation_clauses
! CHECK-SAME: %[[COND_REF:.*]]: !fir.ref<!fir.logical<4>> {fir.bindc_name = "cond"}, %[[NT_REF:.*]]: !fir.ref<i32> {fir.bindc_name = "nt"}
subroutine parallel_do_with_privatisation_clauses(cond,nt)
  ! CHECK: %[[COND_DECL:.*]]:2 = hlfir.declare %[[COND_REF]] dummy_scope %{{[0-9]+}} {uniq_name = "_QFparallel_do_with_privatisation_clausesEcond"} : (!fir.ref<!fir.logical<4>>, !fir.dscope) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
  ! CHECK: %[[NT_DECL:.*]]:2 = hlfir.declare %[[NT_REF]] dummy_scope %{{[0-9]+}} {uniq_name = "_QFparallel_do_with_privatisation_clausesEnt"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  logical :: cond
  integer :: nt
  integer :: i
  ! CHECK:  omp.parallel
  ! CHECK:      %[[PRIVATE_COND_REF:.*]] = fir.alloca !fir.logical<4> {bindc_name = "cond", pinned, uniq_name = "_QFparallel_do_with_privatisation_clausesEcond"}
  ! CHECK:      %[[PRIVATE_COND_DECL:.*]]:2 = hlfir.declare %[[PRIVATE_COND_REF]] {uniq_name = "_QFparallel_do_with_privatisation_clausesEcond"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
  ! CHECK:      %[[PRIVATE_NT_REF:.*]] = fir.alloca i32 {bindc_name = "nt", pinned, uniq_name = "_QFparallel_do_with_privatisation_clausesEnt"}
  ! CHECK:      %[[PRIVATE_NT_DECL:.*]]:2 = hlfir.declare %[[PRIVATE_NT_REF]] {uniq_name = "_QFparallel_do_with_privatisation_clausesEnt"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  ! CHECK:      %[[NT_VAL:.*]] = fir.load %[[NT_DECL]]#0 : !fir.ref<i32>
  ! CHECK:      hlfir.assign %[[NT_VAL]] to %[[PRIVATE_NT_DECL]]#0 : i32, !fir.ref<i32>
  ! CHECK:      %[[WS_LB:.*]] = arith.constant 1 : i32
  ! CHECK:      %[[WS_UB:.*]] = arith.constant 9 : i32
  ! CHECK:      %[[WS_STEP:.*]] = arith.constant 1 : i32
  ! CHECK:      omp.wsloop {
  ! CHECK-NEXT: omp.loop_nest (%[[I:.*]]) : i32 = (%[[WS_LB]]) to (%[[WS_UB]]) inclusive step (%[[WS_STEP]]) {
  !$OMP PARALLEL DO PRIVATE(cond) FIRSTPRIVATE(nt)
  do i=1, 9
  ! CHECK:      fir.store %[[I]] to %[[IV_ADDR:.*]]#1 : !fir.ref<i32>
  ! CHECK:      %[[LOAD_IV:.*]] = fir.load %[[IV_ADDR]]#0 : !fir.ref<i32>
  ! CHECK:      fir.call @_FortranAioOutputInteger32({{.*}}, %[[LOAD_IV]]) {{.*}}: (!fir.ref<i8>, i32) -> i1
  ! CHECK:      %[[PRIVATE_COND_VAL:.*]] = fir.load %[[PRIVATE_COND_DECL]]#0 : !fir.ref<!fir.logical<4>>
  ! CHECK:      %[[PRIVATE_COND_VAL_CVT:.*]] = fir.convert %[[PRIVATE_COND_VAL]] : (!fir.logical<4>) -> i1
  ! CHECK:      fir.call @_FortranAioOutputLogical({{.*}}, %[[PRIVATE_COND_VAL_CVT]]) {{.*}}: (!fir.ref<i8>, i1) -> i1
  ! CHECK:      %[[PRIVATE_NT_VAL:.*]] = fir.load %[[PRIVATE_NT_DECL]]#0 : !fir.ref<i32>
  ! CHECK:      fir.call @_FortranAioOutputInteger32({{.*}}, %[[PRIVATE_NT_VAL]]) {{.*}}: (!fir.ref<i8>, i32) -> i1
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
! CHECK:           %[[NT_DECL:.*]]:2 = hlfir.declare %[[VAL_1]] dummy_scope %{{[0-9]+}} {uniq_name = "_QFparallel_private_doEnt"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           omp.parallel private(@{{.*}} %{{.*}}#0 -> %[[COND_ADDR:.*]], @{{.*firstprivate.*}} %{{.*}}#0 -> %[[NT_PRIV_ADDR:.*]] : {{.*}}) {

! CHECK:             %[[COND_DECL:.*]]:2 = hlfir.declare %[[COND_ADDR]] {uniq_name = "_QFparallel_private_doEcond"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)

! CHECK:             %[[NT_PRIV_DECL:.*]]:2 = hlfir.declare %[[NT_PRIV_ADDR]] {uniq_name = "_QFparallel_private_doEnt"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

! CHECK:             %[[I_PRIV:.*]] = fir.alloca i32 {bindc_name = "i", pinned, {{.*}}}
! CHECK:             %[[I_PRIV_DECL:.*]]:2 = hlfir.declare %[[I_PRIV]] {uniq_name = "_QFparallel_private_doEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

! CHECK:             %[[VAL_7:.*]] = arith.constant 1 : i32
! CHECK:             %[[VAL_8:.*]] = arith.constant 9 : i32
! CHECK:             %[[VAL_9:.*]] = arith.constant 1 : i32
! CHECK:             omp.wsloop {
! CHECK-NEXT:          omp.loop_nest (%[[I:.*]]) : i32 = (%[[VAL_7]]) to (%[[VAL_8]]) inclusive step (%[[VAL_9]]) {
! CHECK:                 fir.store %[[I]] to %[[I_PRIV_DECL]]#1 : !fir.ref<i32>
! CHECK:                 fir.call @_QPfoo(%[[I_PRIV_DECL]]#1, %[[COND_DECL]]#1, %[[NT_PRIV_DECL]]#1) {{.*}}: (!fir.ref<i32>, !fir.ref<!fir.logical<4>>, !fir.ref<i32>) -> ()
! CHECK:                 omp.yield
! CHECK:               }
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
! CHECK:            %[[A_DECL:.*]]:2 = hlfir.declare %[[A_ADDR]] dummy_scope %{{[0-9]+}} {uniq_name = "_QFomp_parallel_multiple_firstprivate_doEa"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:            %[[B_DECL:.*]]:2 = hlfir.declare %[[B_ADDR]] dummy_scope %{{[0-9]+}} {uniq_name = "_QFomp_parallel_multiple_firstprivate_doEb"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           omp.parallel private(@{{.*firstprivate.*}} %{{.*}}#0 -> %[[A_PRIV_ADDR:.*]], @{{.*firstprivate.*}} %{{.*}}#0 -> %[[B_PRIV_ADDR:.*]] : {{.*}}) {

! CHECK:             %[[A_PRIV_DECL:.*]]:2 = hlfir.declare %[[A_PRIV_ADDR]] {uniq_name = "_QFomp_parallel_multiple_firstprivate_doEa"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

! CHECK:             %[[B_PRIV_DECL:.*]]:2 = hlfir.declare %[[B_PRIV_ADDR]] {uniq_name = "_QFomp_parallel_multiple_firstprivate_doEb"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

! CHECK:             %[[I_PRIV_ADDR:.*]] = fir.alloca i32 {bindc_name = "i", pinned, {{.*}}}
! CHECK:             %[[I_PRIV_DECL:.*]]:2 = hlfir.declare %[[I_PRIV_ADDR]] {uniq_name = "_QFomp_parallel_multiple_firstprivate_doEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

! CHECK:             %[[VAL_8:.*]] = arith.constant 1 : i32
! CHECK:             %[[VAL_9:.*]] = arith.constant 10 : i32
! CHECK:             %[[VAL_10:.*]] = arith.constant 1 : i32
! CHECK:             omp.wsloop {
! CHECK-NEXT:          omp.loop_nest (%[[I:.*]]) : i32 = (%[[VAL_8]]) to (%[[VAL_9]]) inclusive step (%[[VAL_10]]) {
! CHECK:                 fir.store %[[I]] to %[[I_PRIV_DECL]]#1 : !fir.ref<i32>
! CHECK:                 fir.call @_QPbar(%[[I_PRIV_DECL]]#1, %[[A_PRIV_DECL]]#1) {{.*}}: (!fir.ref<i32>, !fir.ref<i32>) -> ()
! CHECK:                 omp.yield
! CHECK:               }
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
! CHECK:           %[[NT_DECL:.*]]:2 = hlfir.declare %[[VAL_1]] dummy_scope %{{[0-9]+}} {uniq_name = "_QFparallel_do_privateEnt"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           omp.parallel   {

! CHECK:             %[[COND_PRIV_ADDR:.*]] = fir.alloca !fir.logical<4> {bindc_name = "cond", pinned, uniq_name = "_QFparallel_do_privateEcond"}
! CHECK:             %[[COND_PRIV_DECL:.*]]:2 = hlfir.declare %[[COND_PRIV_ADDR]] {uniq_name = "_QFparallel_do_privateEcond"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)

! CHECK:             %[[NT_PRIV_ADDR:.*]] = fir.alloca i32 {bindc_name = "nt", pinned, uniq_name = "_QFparallel_do_privateEnt"}
! CHECK:             %[[NT_PRIV_DECL:.*]]:2 = hlfir.declare %[[NT_PRIV_ADDR]] {uniq_name = "_QFparallel_do_privateEnt"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:             %[[NT_VAL:.*]] = fir.load %[[NT_DECL]]#0 : !fir.ref<i32>
! CHECK:             hlfir.assign %[[NT_VAL]] to %[[NT_PRIV_DECL]]#0 : i32, !fir.ref<i32>

! CHECK:             %[[I_PRIV_ADDR:.*]] = fir.alloca i32 {bindc_name = "i", pinned, {{.*}}}
! CHECK:             %[[I_PRIV_DECL:.*]]:2 = hlfir.declare %[[I_PRIV_ADDR]] {uniq_name = "_QFparallel_do_privateEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

! CHECK:             %[[VAL_7:.*]] = arith.constant 1 : i32
! CHECK:             %[[VAL_8:.*]] = arith.constant 9 : i32
! CHECK:             %[[VAL_9:.*]] = arith.constant 1 : i32
! CHECK:             omp.wsloop {
! CHECK-NEXT:          omp.loop_nest (%[[I:.*]]) : i32 = (%[[VAL_7]]) to (%[[VAL_8]]) inclusive step (%[[VAL_9]]) {
! CHECK:                 fir.store %[[I]] to %[[I_PRIV_DECL]]#1 : !fir.ref<i32>
! CHECK:                 fir.call @_QPfoo(%[[I_PRIV_DECL]]#1, %[[COND_PRIV_DECL]]#1, %[[NT_PRIV_DECL]]#1) {{.*}}: (!fir.ref<i32>, !fir.ref<!fir.logical<4>>, !fir.ref<i32>) -> ()
! CHECK:                 omp.yield
! CHECK:               }
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
! CHECK:           %[[A_DECL:.*]]:2 = hlfir.declare %[[A_ADDR]] dummy_scope %{{[0-9]+}} {uniq_name = "_QFomp_parallel_do_multiple_firstprivateEa"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[B_DECL:.*]]:2 = hlfir.declare %[[B_ADDR]] dummy_scope %{{[0-9]+}} {uniq_name = "_QFomp_parallel_do_multiple_firstprivateEb"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>
! CHECK:           omp.parallel {

! CHECK:             %[[A_PRIV_ADDR:.*]] = fir.alloca i32 {bindc_name = "a", pinned, uniq_name = "_QFomp_parallel_do_multiple_firstprivateEa"}
! CHECK:             %[[A_PRIV_DECL:.*]]:2 = hlfir.declare %[[A_PRIV_ADDR]] {uniq_name = "_QFomp_parallel_do_multiple_firstprivateEa"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:             %[[A:.*]] = fir.load %[[A_DECL]]#0 : !fir.ref<i32>
! CHECK:             hlfir.assign %[[A]] to %[[A_PRIV_DECL]]#0 : i32, !fir.ref<i32>

! CHECK:             %[[B_PRIV_ADDR:.*]] = fir.alloca i32 {bindc_name = "b", pinned, uniq_name = "_QFomp_parallel_do_multiple_firstprivateEb"}
! CHECK:             %[[B_PRIV_DECL:.*]]:2 = hlfir.declare %[[B_PRIV_ADDR]] {uniq_name = "_QFomp_parallel_do_multiple_firstprivateEb"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:             %[[B:.*]] = fir.load %[[B_DECL]]#0 : !fir.ref<i32>
! CHECK:             hlfir.assign %[[B]] to %[[B_PRIV_DECL]]#0 : i32, !fir.ref<i32>

! CHECK:             %[[I_PRIV_ADDR:.*]] = fir.alloca i32 {bindc_name = "i", pinned, {{.*}}}
! CHECK:             %[[I_PRIV_DECL:.*]]:2 = hlfir.declare %[[I_PRIV_ADDR]] {uniq_name = "_QFomp_parallel_do_multiple_firstprivateEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

! CHECK:             %[[VAL_8:.*]] = arith.constant 1 : i32
! CHECK:             %[[VAL_9:.*]] = arith.constant 10 : i32
! CHECK:             %[[VAL_10:.*]] = arith.constant 1 : i32
! CHECK:             omp.wsloop {
! CHECK-NEXT:         omp.loop_nest (%[[I:.*]]) : i32 = (%[[VAL_8]]) to (%[[VAL_9]]) inclusive step (%[[VAL_10]]) {
! CHECK:                 fir.store %[[I]] to %[[I_PRIV_DECL]]#1 : !fir.ref<i32>
! CHECK:                 fir.call @_QPbar(%[[I_PRIV_DECL]]#1, %[[A_PRIV_DECL]]#1) {{.*}}: (!fir.ref<i32>, !fir.ref<i32>) -> ()
! CHECK:                 omp.yield
! CHECK:               }
! CHECK:             }
! CHECK:             omp.terminator
! CHECK:           }
! CHECK:          return
! CHECK:         }

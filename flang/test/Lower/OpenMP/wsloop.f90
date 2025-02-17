! This test checks lowering of OpenMP DO Directive (Worksharing).

! RUN: bbc -fopenmp -emit-hlfir %s -o - | FileCheck %s

!CHECK-LABEL: func @_QPsimple_loop()
subroutine simple_loop
  integer :: i
  ! CHECK:  omp.parallel
  !$OMP PARALLEL
  ! CHECK:      %[[WS_LB:.*]] = arith.constant 1 : i32
  ! CHECK:      %[[WS_UB:.*]] = arith.constant 9 : i32
  ! CHECK:      %[[WS_STEP:.*]] = arith.constant 1 : i32
  ! CHECK:      omp.wsloop private(@{{.*}} %{{.*}}#0 -> %[[ALLOCA_IV:.*]] : !fir.ref<i32>) {
  ! CHECK-NEXT:   omp.loop_nest (%[[I:.*]]) : i32 = (%[[WS_LB]]) to (%[[WS_UB]]) inclusive step (%[[WS_STEP]]) {
  !$OMP DO
  do i=1, 9
  ! CHECK:          %[[IV_DECL:.*]]:2 = hlfir.declare %[[ALLOCA_IV]] {uniq_name = "_QFsimple_loopEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  ! CHECK:          fir.store %[[I]] to %[[IV_DECL:.*]]#1 : !fir.ref<i32>
  ! CHECK:          %[[LOAD_IV:.*]] = fir.load %[[IV_DECL]]#0 : !fir.ref<i32>
  ! CHECK:          fir.call @_FortranAioOutputInteger32({{.*}}, %[[LOAD_IV]]) {{.*}}: (!fir.ref<i8>, i32) -> i1
    print*, i
  end do
  ! CHECK:          omp.yield
  !$OMP END DO
  ! CHECK:      omp.terminator
  !$OMP END PARALLEL
end subroutine

!CHECK-LABEL: func @_QPsimple_loop_with_step()
subroutine simple_loop_with_step
  integer :: i
  ! CHECK:  omp.parallel
  !$OMP PARALLEL
  ! CHECK:      %[[WS_LB:.*]] = arith.constant 1 : i32
  ! CHECK:      %[[WS_UB:.*]] = arith.constant 9 : i32
  ! CHECK:      %[[WS_STEP:.*]] = arith.constant 2 : i32
  ! CHECK:      omp.wsloop private(@{{.*}} %{{.*}}#0 -> %[[ALLOCA_IV:.*]] : !fir.ref<i32>) {
  ! CHECK-NEXT:   omp.loop_nest (%[[I:.*]]) : i32 = (%[[WS_LB]]) to (%[[WS_UB]]) inclusive step (%[[WS_STEP]]) {
  ! CHECK:          %[[IV_DECL:.*]]:2 = hlfir.declare %[[ALLOCA_IV]] {uniq_name = "_QFsimple_loop_with_stepEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  ! CHECK:          fir.store %[[I]] to %[[IV_DECL]]#1 : !fir.ref<i32>
  ! CHECK:          %[[LOAD_IV:.*]] = fir.load %[[IV_DECL]]#0 : !fir.ref<i32>
  !$OMP DO
  do i=1, 9, 2
  ! CHECK:          fir.call @_FortranAioOutputInteger32({{.*}}, %[[LOAD_IV]]) {{.*}}: (!fir.ref<i8>, i32) -> i1
    print*, i
  end do
  ! CHECK:          omp.yield
  !$OMP END DO
  ! CHECK:      omp.terminator
  !$OMP END PARALLEL
end subroutine

!CHECK-LABEL: func @_QPloop_with_schedule_nowait()
subroutine loop_with_schedule_nowait
  integer :: i
  ! CHECK:  omp.parallel
  !$OMP PARALLEL
  ! CHECK:      %[[WS_LB:.*]] = arith.constant 1 : i32
  ! CHECK:      %[[WS_UB:.*]] = arith.constant 9 : i32
  ! CHECK:      %[[WS_STEP:.*]] = arith.constant 1 : i32
  ! CHECK:      omp.wsloop nowait schedule(runtime) private(@{{.*}} %{{.*}}#0 -> %[[ALLOCA_IV:.*]] : !fir.ref<i32>) {
  ! CHECK-NEXT:   omp.loop_nest (%[[I:.*]]) : i32 = (%[[WS_LB]]) to (%[[WS_UB]]) inclusive step (%[[WS_STEP]]) {
  !$OMP DO SCHEDULE(runtime)
  do i=1, 9
  ! CHECK:          %[[IV_DECL:.*]]:2 = hlfir.declare %[[ALLOCA_IV]] {uniq_name = "_QFloop_with_schedule_nowaitEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  ! CHECK:          fir.store %[[I]] to %[[IV_DECL]]#1 : !fir.ref<i32>
  ! CHECK:          %[[LOAD_IV:.*]] = fir.load %[[IV_DECL]]#0 : !fir.ref<i32>
  ! CHECK:          fir.call @_FortranAioOutputInteger32({{.*}}, %[[LOAD_IV]]) {{.*}}: (!fir.ref<i8>, i32) -> i1
    print*, i
  end do
  ! CHECK:          omp.yield
  !$OMP END DO NOWAIT
  ! CHECK:      omp.terminator
  !$OMP END PARALLEL
end subroutine

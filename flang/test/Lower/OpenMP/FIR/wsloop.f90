! This test checks lowering of OpenMP DO Directive (Worksharing).

! RUN: bbc -fopenmp -emit-fir %s -o - | FileCheck %s

!CHECK-LABEL: func @_QPsimple_loop()
subroutine simple_loop
  integer :: i
  ! CHECK:  omp.parallel
  !$OMP PARALLEL
  ! CHECK:     %[[ALLOCA_IV:.*]] = fir.alloca i32 {{{.*}}, pinned}
  ! CHECK:     %[[WS_LB:.*]] = arith.constant 1 : i32
  ! CHECK:     %[[WS_UB:.*]] = arith.constant 9 : i32
  ! CHECK:     %[[WS_STEP:.*]] = arith.constant 1 : i32
  ! CHECK:     omp.wsloop for (%[[I:.*]]) : i32 = (%[[WS_LB]]) to (%[[WS_UB]]) inclusive step (%[[WS_STEP]])
  !$OMP DO
  do i=1, 9
  ! CHECK:             fir.store %[[I]] to %[[ALLOCA_IV:.*]] : !fir.ref<i32>
  ! CHECK:             %[[LOAD_IV:.*]] = fir.load %[[ALLOCA_IV]] : !fir.ref<i32>
  ! CHECK:    fir.call @_FortranAioOutputInteger32({{.*}}, %[[LOAD_IV]]) {{.*}}: (!fir.ref<i8>, i32) -> i1
    print*, i
  end do
  ! CHECK:       omp.yield
  !$OMP END DO
  ! CHECK:       omp.terminator
  !$OMP END PARALLEL
end subroutine

!CHECK-LABEL: func @_QPsimple_loop_with_step()
subroutine simple_loop_with_step
  integer :: i
  ! CHECK:  omp.parallel
  !$OMP PARALLEL
  ! CHECK:     %[[ALLOCA_IV:.*]] = fir.alloca i32 {{{.*}}, pinned}
  ! CHECK:     %[[WS_LB:.*]] = arith.constant 1 : i32
  ! CHECK:     %[[WS_UB:.*]] = arith.constant 9 : i32
  ! CHECK:     %[[WS_STEP:.*]] = arith.constant 2 : i32
  ! CHECK:     omp.wsloop for (%[[I:.*]]) : i32 = (%[[WS_LB]]) to (%[[WS_UB]]) inclusive step (%[[WS_STEP]])
  ! CHECK:       fir.store %[[I]] to %[[ALLOCA_IV]] : !fir.ref<i32>
  ! CHECK:       %[[LOAD_IV:.*]] = fir.load %[[ALLOCA_IV]] : !fir.ref<i32>
  !$OMP DO
  do i=1, 9, 2
  ! CHECK:    fir.call @_FortranAioOutputInteger32({{.*}}, %[[LOAD_IV]]) {{.*}}: (!fir.ref<i8>, i32) -> i1
    print*, i
  end do
  ! CHECK:       omp.yield
  !$OMP END DO
  ! CHECK:       omp.terminator
  !$OMP END PARALLEL
end subroutine

!CHECK-LABEL: func @_QPloop_with_schedule_nowait()
subroutine loop_with_schedule_nowait
  integer :: i
  ! CHECK:  omp.parallel
  !$OMP PARALLEL
  ! CHECK:     %[[ALLOCA_IV:.*]] = fir.alloca i32 {{{.*}}, pinned}
  ! CHECK:     %[[WS_LB:.*]] = arith.constant 1 : i32
  ! CHECK:     %[[WS_UB:.*]] = arith.constant 9 : i32
  ! CHECK:     %[[WS_STEP:.*]] = arith.constant 1 : i32
  ! CHECK:     omp.wsloop schedule(runtime) nowait for (%[[I:.*]]) : i32 = (%[[WS_LB]]) to (%[[WS_UB]]) inclusive step (%[[WS_STEP]])
  !$OMP DO SCHEDULE(runtime)
  do i=1, 9
  ! CHECK:       fir.store %[[I]] to %[[ALLOCA_IV]] : !fir.ref<i32>
  ! CHECK:       %[[LOAD_IV:.*]] = fir.load %[[ALLOCA_IV]] : !fir.ref<i32>
  ! CHECK:    fir.call @_FortranAioOutputInteger32({{.*}}, %[[LOAD_IV]]) {{.*}}: (!fir.ref<i8>, i32) -> i1
    print*, i
  end do
  ! CHECK:       omp.yield
  !$OMP END DO NOWAIT
  ! CHECK:       omp.terminator
  !$OMP END PARALLEL
end subroutine

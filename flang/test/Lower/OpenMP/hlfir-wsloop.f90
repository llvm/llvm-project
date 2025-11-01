! This test checks lowering of OpenMP DO Directive with HLFIR.

! RUN: bbc -hlfir -fopenmp -emit-fir %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir -flang-experimental-hlfir -fopenmp %s -o - | FileCheck %s

!CHECK-LABEL: func @_QPsimple_loop()
subroutine simple_loop
  integer :: i
  ! CHECK-DAG:     %[[WS_ST:.*]] = arith.constant 1 : i32
  ! CHECK-DAG:     %[[WS_END:.*]] = arith.constant 9 : i32
  ! CHECK:  omp.parallel
  !$OMP PARALLEL
  ! CHECK:         omp.wsloop private(@{{.*}} %{{.*}} -> %[[ALLOCA_IV:.*]] : !fir.ref<i32>) {
  ! CHECK-NEXT:      omp.loop_nest (%[[I:.*]]) : i32 = (%[[WS_ST]]) to (%[[WS_END]]) inclusive step (%[[WS_ST]]) {
  !$OMP DO
  do i=1, 9
  ! CHECK:         %[[IV:.*]]    = fir.declare %[[ALLOCA_IV]] {uniq_name = "_QFsimple_loopEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  ! CHECK:             fir.store %[[I]] to %[[IV:.*]] : !fir.ref<i32>
  ! CHECK:             %[[LOAD_IV:.*]] = fir.load %[[IV]] : !fir.ref<i32>
  ! CHECK:             fir.call @_FortranAioOutputInteger32({{.*}}, %[[LOAD_IV]]) {{.*}}: (!fir.ref<i8>, i32) -> i1
    print*, i
  end do
  ! CHECK:             omp.yield
  !$OMP END DO
  ! CHECK:         omp.terminator
  !$OMP END PARALLEL
end subroutine

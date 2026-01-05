! RUN: bbc --use-desc-for-alloc=false -fopenmp -fopenmp-version=45 -emit-hlfir %s -o - \
! RUN: | FileCheck %s --check-prefix=FIRDialect

!CHECK-LABEL: func @_QPsimd_loop_1()
subroutine simd_loop_1
  integer :: i
  real, allocatable :: r;

  ! FIRDialect:     %[[LB:.*]] = arith.constant 1 : i32
  ! FIRDialect:     %[[UB:.*]] = arith.constant 9 : i32
  ! FIRDialect:     %[[STEP:.*]] = arith.constant 1 : i32

  ! FIRDialect: omp.simd linear({{.*}} = %[[STEP]] : !fir.ref<i32>) private({{.*}}) {
  ! FIRDialect-NEXT: omp.loop_nest (%[[I:.*]]) : i32 = (%[[LB]]) to (%[[UB]]) inclusive step (%[[STEP]]) {
  !$OMP SIMD PRIVATE(r)
  do i=1, 9
  ! FIRDialect:     hlfir.assign %[[I]] to %[[LOCAL:.*]]#0 : i32, !fir.ref<i32>
  ! FIRDialect:     %[[LOAD_IV:.*]] = fir.load %[[LOCAL]]#0 : !fir.ref<i32>
  ! FIRDialect:     fir.call @_FortranAioOutputInteger32({{.*}}, %[[LOAD_IV]]) {{.*}}: (!fir.ref<i8>, i32) -> i1
    print*, i
  end do
  !$OMP END SIMD
  ! FIRDialect:     omp.yield

end subroutine

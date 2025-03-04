! Mark as xfail for now until we upstream the relevant part. This is just for
! demo purposes at this point. Upstreaming this is the next step.
! XFAIL: *

! Tests mapping of a basic `do concurrent` loop to `!$omp parallel do`.

! RUN: %flang_fc1 -emit-hlfir -fopenmp -fdo-concurrent-to-openmp=host %s -o - \
! RUN:   | FileCheck %s
! RUN: bbc -emit-hlfir -fopenmp -fdo-concurrent-to-openmp=host %s -o - \
! RUN:   | FileCheck %s
 
! CHECK-LABEL: do_concurrent_basic
program do_concurrent_basic
    ! CHECK: %[[ARR:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFEa"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)

    implicit none
    integer :: a(10)
    integer :: i

    ! CHECK-NOT: fir.do_loop

    ! CHECK: omp.parallel {

    ! CHECK-NEXT: %[[ITER_VAR:.*]] = fir.alloca i32 {bindc_name = "i"}
    ! CHECK-NEXT: %[[BINDING:.*]]:2 = hlfir.declare %[[ITER_VAR]] {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

    ! CHECK: %[[C1:.*]] = arith.constant 1 : i32
    ! CHECK: %[[LB:.*]] = fir.convert %[[C1]] : (i32) -> index
    ! CHECK: %[[C10:.*]] = arith.constant 10 : i32
    ! CHECK: %[[UB:.*]] = fir.convert %[[C10]] : (i32) -> index
    ! CHECK: %[[STEP:.*]] = arith.constant 1 : index

    ! CHECK: omp.wsloop {
    ! CHECK-NEXT: omp.loop_nest (%[[ARG0:.*]]) : index = (%[[LB]]) to (%[[UB]]) inclusive step (%[[STEP]]) {
    ! CHECK-NEXT: %[[IV_IDX:.*]] = fir.convert %[[ARG0]] : (index) -> i32
    ! CHECK-NEXT: fir.store %[[IV_IDX]] to %[[BINDING]]#1 : !fir.ref<i32>
    ! CHECK-NEXT: %[[IV_VAL1:.*]] = fir.load %[[BINDING]]#0 : !fir.ref<i32>
    ! CHECK-NEXT: %[[IV_VAL2:.*]] = fir.load %[[BINDING]]#0 : !fir.ref<i32>
    ! CHECK-NEXT: %[[IV_VAL_I64:.*]] = fir.convert %[[IV_VAL2]] : (i32) -> i64
    ! CHECK-NEXT: %[[ARR_ACCESS:.*]] = hlfir.designate %[[ARR]]#0 (%[[IV_VAL_I64]])  : (!fir.ref<!fir.array<10xi32>>, i64) -> !fir.ref<i32>
    ! CHECK-NEXT: hlfir.assign %[[IV_VAL1]] to %[[ARR_ACCESS]] : i32, !fir.ref<i32>
    ! CHECK-NEXT: omp.yield
    ! CHECK-NEXT: }
    ! CHECK-NEXT: }

    ! CHECK-NEXT: omp.terminator
    ! CHECK-NEXT: }
    do concurrent (i=1:10)
        a(i) = i
    end do

    ! CHECK-NOT: fir.do_loop
end program do_concurrent_basic

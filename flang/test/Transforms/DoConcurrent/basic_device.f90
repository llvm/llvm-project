! Tests mapping of a basic `do concurrent` loop to
! `!$omp target teams distribute parallel do`.

! RUN: %flang_fc1 -emit-hlfir -fopenmp -fdo-concurrent-to-openmp=device %s -o - \
! RUN:   | FileCheck %s
! RUN: bbc -emit-hlfir -fopenmp -fdo-concurrent-to-openmp=device %s -o - \
! RUN:   | FileCheck %s

program do_concurrent_basic
    implicit none
    integer :: a(10)
    integer :: i

    ! CHECK: %[[I_ORIG_ALLOC:.*]] = fir.alloca i32 {bindc_name = "i"}
    ! CHECK: %[[I_ORIG_DECL:.*]]:2 = hlfir.declare %[[I_ORIG_ALLOC]]

    ! CHECK: %[[A_ADDR:.*]] = fir.address_of(@_QFEa)
    ! CHECK: %[[A_SHAPE:.*]] = fir.shape %[[A_EXTENT:.*]] : (index) -> !fir.shape<1>
    ! CHECK: %[[A_ORIG_DECL:.*]]:2 = hlfir.declare %[[A_ADDR]](%[[A_SHAPE]])

    ! CHECK-NOT: fir.do_loop

    ! CHECK: %[[C1:.*]] = arith.constant 1 : i32
    ! CHECK: %[[HOST_LB:.*]] = fir.convert %[[C1]] : (i32) -> index
    ! CHECK: %[[C10:.*]] = arith.constant 10 : i32
    ! CHECK: %[[HOST_UB:.*]] = fir.convert %[[C10]] : (i32) -> index
    ! CHECK: %[[HOST_STEP:.*]] = arith.constant 1 : index

    ! CHECK: %[[I_MAP_INFO:.*]] = omp.map.info var_ptr(%[[I_ORIG_DECL]]#1
    ! CHECK: %[[C0:.*]] = arith.constant 0 : index
    ! CHECK: %[[UPPER_BOUND:.*]] = arith.subi %[[A_EXTENT]], %{{c1.*}} : index

    ! CHECK: %[[A_BOUNDS:.*]] = omp.map.bounds lower_bound(%[[C0]] : index)
    ! CHECK-SAME: upper_bound(%[[UPPER_BOUND]] : index)
    ! CHECK-SAME: extent(%[[A_EXTENT]] : index)

    ! CHECK: %[[A_MAP_INFO:.*]] = omp.map.info var_ptr(%[[A_ORIG_DECL]]#1 : {{[^(]+}})
    ! CHECK-SAME: map_clauses(implicit, tofrom) capture(ByRef) bounds(%[[A_BOUNDS]])

    ! CHECK: omp.target
    ! CHECK-SAME: host_eval(%[[HOST_LB]] -> %[[LB:[[:alnum:]]+]], %[[HOST_UB]] -> %[[UB:[[:alnum:]]+]], %[[HOST_STEP]] -> %[[STEP:[[:alnum:]]+]] : index, index, index)
    ! CHECK-SAME: map_entries(
    ! CHECK-SAME:     %{{[[:alnum:]]+}} -> %{{[^,]+}},
    ! CHECK-SAME:     %{{[[:alnum:]]+}} -> %{{[^,]+}},
    ! CHECK-SAME:     %{{[[:alnum:]]+}} -> %{{[^,]+}},
    ! CHECK-SAME:     %[[I_MAP_INFO]] -> %[[I_ARG:[[:alnum:]]+]],
    ! CHECK-SAME:             %[[A_MAP_INFO]] -> %[[A_ARG:.[[:alnum:]]+]]

    ! CHECK: %[[A_DEV_DECL:.*]]:2 = hlfir.declare %[[A_ARG]]
    ! CHECK: omp.teams {
    ! CHECK-NEXT: omp.parallel {

    ! CHECK-NEXT: %[[ITER_VAR:.*]] = fir.alloca i32 {bindc_name = "i"}
    ! CHECK-NEXT: %[[BINDING:.*]]:2 = hlfir.declare %[[ITER_VAR]] {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

    ! CHECK-NEXT: omp.distribute {
    ! CHECK-NEXT: omp.wsloop {

    ! CHECK-NEXT: omp.loop_nest (%[[ARG0:.*]]) : index = (%[[LB]]) to (%[[UB]]) inclusive step (%[[STEP]]) {
    ! CHECK-NEXT: %[[IV_IDX:.*]] = fir.convert %[[ARG0]] : (index) -> i32
    ! CHECK-NEXT: fir.store %[[IV_IDX]] to %[[BINDING]]#0 : !fir.ref<i32>
    ! CHECK-NEXT: %[[IV_VAL1:.*]] = fir.load %[[BINDING]]#0 : !fir.ref<i32>
    ! CHECK-NEXT: %[[IV_VAL2:.*]] = fir.load %[[BINDING]]#0 : !fir.ref<i32>
    ! CHECK-NEXT: %[[IV_VAL_I64:.*]] = fir.convert %[[IV_VAL2]] : (i32) -> i64
    ! CHECK-NEXT: %[[ARR_ACCESS:.*]] = hlfir.designate %[[A_DEV_DECL]]#0 (%[[IV_VAL_I64]])  : (!fir.ref<!fir.array<10xi32>>, i64) -> !fir.ref<i32>
    ! CHECK-NEXT: hlfir.assign %[[IV_VAL1]] to %[[ARR_ACCESS]] : i32, !fir.ref<i32>
    ! CHECK-NEXT: omp.yield
    ! CHECK-NEXT: }

    ! CHECK-NEXT: } {omp.composite}
    ! CHECK-NEXT: } {omp.composite}
    ! CHECK-NEXT: omp.terminator
    ! CHECK-NEXT: } {omp.composite}
    ! CHECK-NEXT: omp.terminator
    ! CHECK-NEXT: }
    ! CHECK-NEXT: omp.terminator
    ! CHECK-NEXT: }
    do concurrent (i=1:10)
        a(i) = i
    end do

    ! CHECK-NOT: fir.do_loop
end program do_concurrent_basic

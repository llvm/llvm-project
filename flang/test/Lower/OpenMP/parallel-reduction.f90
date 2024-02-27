! RUN: bbc -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s

!CHECK:  omp.reduction.declare @[[REDUCTION_DECLARE:[_a-z0-9]+]] : i32 init {
!CHECK:  ^bb0(%{{.*}}: i32):
!CHECK:    %[[I0:[_a-z0-9]+]] = arith.constant 0 : i32
!CHECK:    omp.yield(%[[I0]] : i32)
!CHECK:  } combiner {
!CHECK:  ^bb0(%[[C0:[_a-z0-9]+]]: i32, %[[C1:[_a-z0-9]+]]: i32):
!CHECK:    %[[CR:[_a-z0-9]+]] = arith.addi %[[C0]], %[[C1]] : i32
!CHECK:    omp.yield(%[[CR]] : i32)
!CHECK:  }
!CHECK:  func.func @_QQmain() attributes {fir.bindc_name = "mn"} {
!CHECK:    %[[RED_ACCUM_REF:[_a-z0-9]+]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFEi"}
!CHECK:    %[[RED_ACCUM_DECL:[_a-z0-9]+]]:2 = hlfir.declare %[[RED_ACCUM_REF]] {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:    %[[C0:[_a-z0-9]+]] = arith.constant 0 : i32
!CHECK:    hlfir.assign %[[C0]] to %[[RED_ACCUM_DECL]]#0 : i32, !fir.ref<i32>
!CHECK:    omp.parallel reduction(@[[REDUCTION_DECLARE]] %[[RED_ACCUM_DECL]]#0 -> %[[PRIVATE_RED:[a-z0-9]+]] : !fir.ref<i32>) {
!CHECK:      %[[PRIVATE_DECL:[_a-z0-9]+]]:2 = hlfir.declare %[[PRIVATE_RED]] {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:      %[[C1:[_a-z0-9]+]] = arith.constant 1 : i32
!CHECK:      hlfir.assign %[[C1]] to %[[PRIVATE_DECL]]#0 : i32, !fir.ref<i32>
!CHECK:      omp.terminator
!CHECK:    }
!CHECK:    %[[RED_ACCUM_VAL:[_a-z0-9]+]] = fir.load %[[RED_ACCUM_DECL]]#0 : !fir.ref<i32>
!CHECK:    {{.*}} = fir.call @_FortranAioOutputInteger32(%{{.*}}, %[[RED_ACCUM_VAL]]) fastmath<contract> : (!fir.ref<i8>, i32) -> i1
!CHECK:    return
!CHECK:  }

program mn
    integer :: i
    i = 0

    !$omp parallel reduction(+:i)
      i = 1
    !$omp end parallel

    print *, i
end program

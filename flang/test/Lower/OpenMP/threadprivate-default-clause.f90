! REQUIRES: openmp_runtime

! Simple test for lowering of OpenMP Threadprivate Directive with HLFIR.

!RUN: %flang_fc1  -emit-hlfir %openmp_flags %s -o - | FileCheck %s

!CHECK-LABEL: func.func @_QPsub1() {
!CHECK:     %[[A:.*]] = fir.address_of(@_QFsub1Ea) : !fir.ref<i32>
!CHECK:     %[[A_DECL:.*]]:2 = hlfir.declare %[[A]]  {uniq_name = "_QFsub1Ea"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:     %[[A_TP0:.*]] = omp.threadprivate %[[A_DECL]]#0 : !fir.ref<i32> -> !fir.ref<i32>
!CHECK:     %[[A_CVT:.*]]:2 = hlfir.declare %[[A_TP0]] {uniq_name = "_QFsub1Ea"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:     omp.parallel {
!CHECK:       %[[A_TP:.*]] = omp.threadprivate %[[A_DECL]]#0 : !fir.ref<i32> -> !fir.ref<i32>
!CHECK:       %[[A_TP_DECL:.*]]:2 = hlfir.declare %[[A_TP]] {uniq_name = "_QFsub1Ea"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:       %[[TID:.*]] = fir.call @omp_get_thread_num() proc_attrs<bind_c> fastmath<contract> : () -> i32
!CHECK:       hlfir.assign %[[TID]] to %[[A_TP_DECL]]#0 : i32, !fir.ref<i32>
!CHECK:       omp.terminator
!CHECK:     }

subroutine sub1()
   use omp_lib
   integer, save :: a
   !$omp threadprivate(a)
   !$omp parallel default(none)
   a = omp_get_thread_num()
   !$omp end parallel
end subroutine

!CHECK-LABEL: func.func @_QPsub2() {
!CHECK:    %[[BLK:.*]] = fir.address_of(@blk_) : !fir.ref<!fir.array<4xi8>>
!CHECK:    %c0 = arith.constant 0 : index
!CHECK:    %[[A_ADDR:.*]] = fir.coordinate_of %[[BLK]], %c0 : (!fir.ref<!fir.array<4xi8>>, index) -> !fir.ref<i8>
!CHECK:    %[[A_CVT:.*]] = fir.convert %[[A_ADDR]] : (!fir.ref<i8>) -> !fir.ref<i32>
!CHECK:    %[[A_DECL:.*]]:2 = hlfir.declare %[[A_CVT]] storage(%[[BLK]][0]) {uniq_name = "_QFsub2Ea"} : (!fir.ref<i32>, !fir.ref<!fir.array<4xi8>>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:    %[[A_TP0:.*]] = omp.threadprivate %[[BLK]] : !fir.ref<!fir.array<4xi8>> -> !fir.ref<!fir.array<4xi8>>
!CHECK:    %c0_0 = arith.constant 0 : index
!CHECK:    %[[A_TP0_ADDR:.*]] = fir.coordinate_of %[[A_TP0]], %c0_0 : (!fir.ref<!fir.array<4xi8>>, index) -> !fir.ref<i8>
!CHECK:    %[[A_TP0_ADDR_CVT:.*]] = fir.convert %[[A_TP0_ADDR]] : (!fir.ref<i8>) -> !fir.ref<i32>
!CHECK:    %[[A_TP0_DECL:.*]]:2 = hlfir.declare %[[A_TP0_ADDR_CVT]] storage(%[[A_TP0]][0]) {uniq_name = "_QFsub2Ea"} : (!fir.ref<i32>, !fir.ref<!fir.array<4xi8>>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:    omp.parallel {
!CHECK:       %[[BLK_TP:.*]] = omp.threadprivate %[[BLK]] : !fir.ref<!fir.array<4xi8>> -> !fir.ref<!fir.array<4xi8>>
!CHECK:       %c0_1 = arith.constant 0 : index
!CHECK:       %[[A_TP_ADDR:.*]] = fir.coordinate_of %[[BLK_TP]], %c0_1 : (!fir.ref<!fir.array<4xi8>>, index) -> !fir.ref<i8>
!CHECK:       %[[A_TP_ADDR_CVT:.*]] = fir.convert %[[A_TP_ADDR]] : (!fir.ref<i8>) -> !fir.ref<i32>
!CHECK:       %[[A_TP_DECL:.*]]:2 = hlfir.declare %[[A_TP_ADDR_CVT]] storage(%[[BLK_TP]][0]) {uniq_name = "_QFsub2Ea"} : (!fir.ref<i32>, !fir.ref<!fir.array<4xi8>>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:       %[[TID:.*]] = fir.call @omp_get_thread_num() proc_attrs<bind_c> fastmath<contract> : () -> i32
!CHECK:       hlfir.assign %[[TID]] to %[[A_TP_DECL]]#0 : i32, !fir.ref<i32>
!CHECK:       omp.terminator
!CHECK:     }

subroutine sub2()
   use omp_lib
   integer :: a
   common/blk/a
   !$omp threadprivate(/blk/)
   !$omp parallel default(none)
   a = omp_get_thread_num()
   !$omp end parallel
end subroutine

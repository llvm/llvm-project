! Simple test for lowering of OpenMP Threadprivate Directive with HLFIR.
! Test for common block.

!RUN: %flang_fc1 -flang-experimental-hlfir -emit-hlfir -fopenmp %s -o - | FileCheck %s
!RUN: bbc -hlfir -emit-hlfir -fopenmp %s -o - | FileCheck %s


!CHECK: %[[CBLK_ADDR:.*]] = fir.address_of(@blk_) : !fir.ref<!fir.array<4xi8>>
!CHECK: {{.*}} = omp.threadprivate %[[CBLK_ADDR]] : !fir.ref<!fir.array<4xi8>> -> !fir.ref<!fir.array<4xi8>>
!CHECK: omp.parallel   {
!CHECK:   %[[TP_PARALLEL:.*]] = omp.threadprivate %[[CBLK_ADDR]] : !fir.ref<!fir.array<4xi8>> -> !fir.ref<!fir.array<4xi8>>
!CHECK:   %[[TP_PARALLEL_ADDR:.*]] = fir.convert %[[TP_PARALLEL]] : (!fir.ref<!fir.array<4xi8>>) -> !fir.ref<!fir.array<?xi8>>
!CHECK:   %[[A_ADDR:.*]] = fir.coordinate_of %[[TP_PARALLEL_ADDR]], %c0_1 : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
!CHECK:   %[[A_ADDR_CVT:.*]] = fir.convert %[[A_ADDR]] : (!fir.ref<i8>) -> !fir.ref<i32>
!CHECK:   %[[A_DECL:.*]]:2 = hlfir.declare %[[A_ADDR_CVT]] {uniq_name = "_QFsub_commonblockEa"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:   %[[A_VAL:.*]] = fir.load %[[A_DECL]]#0 : !fir.ref<i32>
!CHECK:   {{.*}} = fir.call @_FortranAioOutputInteger32({{.*}}, %[[A_VAL]]) fastmath<contract> : (!fir.ref<i8>, i32) -> i1
!CHECK:   omp.terminator
!CHECK: }

subroutine sub_commonblock()
  integer:: a
  common /blk/ a
  !$omp threadprivate(/blk/)

  !$omp parallel
    print *, a
  !$omp end parallel
end

! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s
! RUN: bbc -fopenmp -emit-hlfir %s -o - | FileCheck %s

! CHECK: %[[V0:[0-9]+]] = fir.alloca !fir.type<_QFfooTt0{a0:i32,a1:i32}> {bindc_name = "a", uniq_name = "_QFfooEa"}
! CHECK: %[[V1:[0-9]+]]:2 = hlfir.declare %[[V0]] {uniq_name = "_QFfooEa"} : (!fir.ref<!fir.type<_QFfooTt0{a0:i32,a1:i32}>>) -> (!fir.ref<!fir.type<_QFfooTt0{a0:i32,a1:i32}>>, !fir.ref<!fir.type<_QFfooTt0{a0:i32,a1:i32}>>)
! CHECK: %[[V2:[0-9]+]] = hlfir.designate %[[V1]]#0{"a1"}   : (!fir.ref<!fir.type<_QFfooTt0{a0:i32,a1:i32}>>) -> !fir.ref<i32>
! CHECK: %[[V3:[0-9]+]] = omp.map_info var_ptr(%[[V2]] : !fir.ref<i32>, i32) map_clauses(tofrom) capture(ByRef) -> !fir.ref<i32> {name = "a%a1"}
! CHECK: %[[V4:[0-9]+]] = omp.map_info var_ptr(%[[V1]]#1 : !fir.ref<!fir.type<_QFfooTt0{a0:i32,a1:i32}>>, !fir.type<_QFfooTt0{a0:i32,a1:i32}>) map_clauses(implicit, tofrom) capture(ByRef) -> !fir.ref<!fir.type<_QFfooTt0{a0:i32,a1:i32}>> {name = "a"}
! CHECK: omp.target map_entries(%[[V3]] -> %arg0, %[[V4]] -> %arg1 : !fir.ref<i32>, !fir.ref<!fir.type<_QFfooTt0{a0:i32,a1:i32}>>) {
! CHECK: ^bb0(%arg0: !fir.ref<i32>, %arg1: !fir.ref<!fir.type<_QFfooTt0{a0:i32,a1:i32}>>):
! CHECK:   %[[V5:[0-9]+]]:2 = hlfir.declare %arg1 {uniq_name = "_QFfooEa"} : (!fir.ref<!fir.type<_QFfooTt0{a0:i32,a1:i32}>>) -> (!fir.ref<!fir.type<_QFfooTt0{a0:i32,a1:i32}>>, !fir.ref<!fir.type<_QFfooTt0{a0:i32,a1:i32}>>)
! CHECK:   %c0_i32 = arith.constant 0 : i32
! CHECK:   %[[V6:[0-9]+]] = hlfir.designate %[[V5]]#0{"a1"}   : (!fir.ref<!fir.type<_QFfooTt0{a0:i32,a1:i32}>>) -> !fir.ref<i32>
! CHECK:   hlfir.assign %c0_i32 to %[[V6]] : i32, !fir.ref<i32>
! CHECK:   omp.terminator
! CHECK: }

subroutine foo()
  implicit none

  type t0
    integer :: a0, a1
  end type

  type(t0) :: a

  !$omp target map(a%a1)
  a%a1 = 0
  !$omp end target
end


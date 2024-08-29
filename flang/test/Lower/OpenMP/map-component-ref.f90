! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s
! RUN: bbc -fopenmp -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPfoo1
! CHECK: %[[V0:[0-9]+]] = fir.alloca !fir.type<_QFfoo1Tt0{a0:i32,a1:i32}> {bindc_name = "a", uniq_name = "_QFfoo1Ea"}
! CHECK: %[[V1:[0-9]+]]:2 = hlfir.declare %[[V0]] {uniq_name = "_QFfoo1Ea"} : (!fir.ref<!fir.type<_QFfoo1Tt0{a0:i32,a1:i32}>>) -> (!fir.ref<!fir.type<_QFfoo1Tt0{a0:i32,a1:i32}>>, !fir.ref<!fir.type<_QFfoo1Tt0{a0:i32,a1:i32}>>)
! CHECK: %[[V2:[0-9]+]] = hlfir.designate %[[V1]]#0{"a1"}   : (!fir.ref<!fir.type<_QFfoo1Tt0{a0:i32,a1:i32}>>) -> !fir.ref<i32>
! CHECK: %[[V3:[0-9]+]] = omp.map.info var_ptr(%[[V2]] : !fir.ref<i32>, i32) map_clauses(tofrom) capture(ByRef) -> !fir.ref<i32> {name = "a%a1"}
! CHECK: %[[V4:[0-9]+]] = omp.map.info var_ptr(%[[V1]]#1 : !fir.ref<!fir.type<_QFfoo1Tt0{a0:i32,a1:i32}>>, !fir.type<_QFfoo1Tt0{a0:i32,a1:i32}>) map_clauses(tofrom) capture(ByRef) members(%[[V3]] : [1] : !fir.ref<i32>) -> !fir.ref<!fir.type<_QFfoo1Tt0{a0:i32,a1:i32}>> {name = "a", partial_map = true}
! CHECK: omp.target map_entries(%[[V3]] -> %arg0, %[[V4]] -> %arg1 : !fir.ref<i32>, !fir.ref<!fir.type<_QFfoo1Tt0{a0:i32,a1:i32}>>) {
! CHECK: ^bb0(%arg0: !fir.ref<i32>, %arg1: !fir.ref<!fir.type<_QFfoo1Tt0{a0:i32,a1:i32}>>):
! CHECK:   %[[V5:[0-9]+]]:2 = hlfir.declare %arg1 {uniq_name = "_QFfoo1Ea"} : (!fir.ref<!fir.type<_QFfoo1Tt0{a0:i32,a1:i32}>>) -> (!fir.ref<!fir.type<_QFfoo1Tt0{a0:i32,a1:i32}>>, !fir.ref<!fir.type<_QFfoo1Tt0{a0:i32,a1:i32}>>)
! CHECK:   %c0_i32 = arith.constant 0 : i32
! CHECK:   %[[V6:[0-9]+]] = hlfir.designate %[[V5]]#0{"a1"}   : (!fir.ref<!fir.type<_QFfoo1Tt0{a0:i32,a1:i32}>>) -> !fir.ref<i32>
! CHECK:   hlfir.assign %c0_i32 to %[[V6]] : i32, !fir.ref<i32>
! CHECK:   omp.terminator
! CHECK: }

subroutine foo1()
  implicit none

  type t0
    integer :: a0, a1
  end type

  type(t0) :: a

  !$omp target map(a%a1)
  a%a1 = 0
  !$omp end target
end


! CHECK-LABEL: func.func @_QPfoo2
! CHECK-DAG: omp.map.info var_ptr(%{{[0-9]+}} : {{.*}} map_clauses(to) capture(ByRef) bounds(%{{[0-9]+}}) -> {{.*}} {name = "t%b(1_8)%a(1)"}
! CHECK-DAG: omp.map.info var_ptr(%{{[0-9]+}} : {{.*}} map_clauses(from) capture(ByRef) bounds(%{{[0-9]+}}) -> {{.*}} {name = "u%b(1_8)%a(1)"}
subroutine foo2()
  implicit none

  type t0
    integer :: a(10)
  end type

  type t1
    type(t0) :: b(10)
  end type

  type(t1) :: t, u

!$omp target map(to: t%b(1)%a(1)) map(from: u%b(1)%a(1))
  t%b(1)%a(1) = u%b(1)%a(1)
!$omp end target

end

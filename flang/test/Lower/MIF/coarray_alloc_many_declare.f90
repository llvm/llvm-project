! RUN: %flang_fc1 -emit-hlfir -fcoarray %s -o - | FileCheck %s

program p
  integer, allocatable :: a[:]
  allocate(a[*])
  call inner
contains
  subroutine inner
    a = 1
  end subroutine
end program p

!CHECK-LABEL: @_QQmain()
!CHECK:       %[[VAL_0:.*]] = fir.alloca !fir.array<0xi64>
!CHECK-NEXT:  %[[VAL_1:.*]] = fir.alloca !fir.array<1xi64>
!CHECK-NEXT:  %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
!CHECK-NEXT:  %[[VAL_3:.*]] = fir.address_of(@_QFEa) : !fir.ref<!fir.box<!fir.heap<i32>, corank:1>>
!CHECK-NEXT:  %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_3]] {fortran_attrs = #fir.var_attrs<allocatable, internal_assoc>, uniq_name = "_QFEa"} : (!fir.ref<!fir.box<!fir.heap<i32>, corank:1>>) -> (!fir.ref<!fir.box<!fir.heap<i32>, corank:1>>, !fir.ref<!fir.box<!fir.heap<i32>, corank:1>>)
!CHECK-NEXT:  %[[VAL_5:.*]] = fir.absent !fir.box<none>
!CHECK-NEXT:  %c1_i64 = arith.constant 1 : i64
!CHECK-NEXT:  %c0 = arith.constant 0 : index
!CHECK-NEXT:  %[[VAL_6:.*]] = fir.coordinate_of %[[VAL_1]], %c0 : (!fir.ref<!fir.array<1xi64>>, index) -> !fir.ref<i64>
!CHECK-NEXT:  fir.store %c1_i64 to %[[VAL_6]] : !fir.ref<i64>
!CHECK-NEXT:  %[[VAL_7:.*]] = fir.embox %[[VAL_1]] : (!fir.ref<!fir.array<1xi64>>) -> !fir.box<!fir.array<1xi64>>
!CHECK-NEXT:  %[[VAL_8:.*]] = fir.embox %[[VAL_0]] : (!fir.ref<!fir.array<0xi64>>) -> !fir.box<!fir.array<0xi64>>
!CHECK-NEXT:  mif.alloc_coarray %[[VAL_4]]#0 lcobounds %[[VAL_7]] ucobounds %[[VAL_8]] errmsg %[[VAL_5]] {uniq_name = "_QFEa"} : (!fir.ref<!fir.box<!fir.heap<i32>, corank:1>>, !fir.box<!fir.array<1xi64>>, !fir.box<!fir.array<0xi64>>, !fir.box<none>) -> ()
!CHECK-NEXT:  fir.call @_QFPinner() fastmath<contract> : () -> ()

!CHECK-LABEL: func.func private @_QFPinner() attributes {fir.host_symbol = @_QQmain, llvm.linkage = #llvm.linkage<internal>}
!CHECK:       %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
!CHECK-NEXT:  %[[VAL_1:.*]] = fir.address_of(@_QFEa) : !fir.ref<!fir.box<!fir.heap<i32>, corank:1>>
!CHECK-NEXT:  %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFEa"} : (!fir.ref<!fir.box<!fir.heap<i32>, corank:1>>) -> (!fir.ref<!fir.box<!fir.heap<i32>, corank:1>>, !fir.ref<!fir.box<!fir.heap<i32>, corank:1>>)
!CHECK-NEXT:  %c1_i32 = arith.constant 1 : i32
!CHECK-NEXT:  hlfir.assign %c1_i32 to %[[VAL_2:.*]]#0 realloc : i32, !fir.ref<!fir.box<!fir.heap<i32>, corank:1>>

!CHECK: fir.global internal @_QFEa : !fir.box<!fir.heap<i32>, corank:1>

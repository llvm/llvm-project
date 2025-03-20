! RUN: bbc %s -o - | FileCheck %s

program p
  integer,volatile::i,arr(10)
  integer,volatile,target::tgt(10)
  integer,volatile,pointer,dimension(:)::ptr
  ptr => tgt
  i=0
  arr=1
  call d(arr)
  call e(arr)
  call f(arr)
  call g(ptr)
contains
  subroutine d(arr)
    integer,volatile::arr(10)
  end subroutine
  subroutine e(arr)
    integer,volatile,dimension(:)::arr
  end subroutine
  subroutine f(arr)
    integer,volatile,dimension(10)::arr
  end subroutine
  subroutine g(arr)
    integer,volatile,dimension(:),pointer::arr
  end subroutine
end program

! CHECK-LABEL:   func.func @_QQmain() attributes {{.+}} {
! CHECK:           %[[VAL_0:.*]] = arith.constant 1 : i32
! CHECK:           %[[VAL_1:.*]] = arith.constant 0 : i32
! CHECK:           %[[VAL_2:.*]] = arith.constant 10 : index
! CHECK:           %[[VAL_3:.*]] = fir.address_of(@_QFEarr) : !fir.ref<!fir.array<10xi32>>
! CHECK:           %[[VAL_4:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_5:.*]] = fir.volatile_cast %[[VAL_3]] : (!fir.ref<!fir.array<10xi32>>) -> !fir.ref<!fir.array<10xi32>, volatile>
! CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_5]](%[[VAL_4]]) {{.+}} : (!fir.ref<!fir.array<10xi32>, volatile>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>, volatile>, !fir.ref<!fir.array<10xi32>, volatile>)
! CHECK:           %[[VAL_7:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFEi"}
! CHECK:           %[[VAL_8:.*]] = fir.volatile_cast %[[VAL_7]] : (!fir.ref<i32>) -> !fir.ref<i32, volatile>
! CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_8]] {{.+}} : (!fir.ref<i32, volatile>) -> (!fir.ref<i32, volatile>, !fir.ref<i32, volatile>)
! CHECK:           %[[VAL_10:.*]] = fir.address_of(@_QFEptr) : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:           %[[VAL_11:.*]] = fir.volatile_cast %[[VAL_10]] : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>
! CHECK:           %[[VAL_12:.*]]:2 = hlfir.declare %[[VAL_11]] {{.+}} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>)
! CHECK:           %[[VAL_13:.*]] = fir.address_of(@_QFEtgt) : !fir.ref<!fir.array<10xi32>>
! CHECK:           %[[VAL_14:.*]] = fir.volatile_cast %[[VAL_13]] : (!fir.ref<!fir.array<10xi32>>) -> !fir.ref<!fir.array<10xi32>, volatile>
! CHECK:           %[[VAL_15:.*]]:2 = hlfir.declare %[[VAL_14]](%[[VAL_4]]) {{.+}} : (!fir.ref<!fir.array<10xi32>, volatile>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>, volatile>, !fir.ref<!fir.array<10xi32>, volatile>)
! CHECK:           %[[VAL_16:.*]] = fir.embox %[[VAL_15]]#0(%[[VAL_4]]) : (!fir.ref<!fir.array<10xi32>, volatile>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>
! CHECK:           fir.store %[[VAL_16]] to %[[VAL_12]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>
! CHECK:           hlfir.assign %[[VAL_1]] to %[[VAL_9]]#0 : i32, !fir.ref<i32, volatile>
! CHECK:           hlfir.assign %[[VAL_0]] to %[[VAL_6]]#0 : i32, !fir.ref<!fir.array<10xi32>, volatile>
! CHECK:           %[[VAL_17:.*]] = fir.volatile_cast %[[VAL_6]]#0 : (!fir.ref<!fir.array<10xi32>, volatile>) -> !fir.ref<!fir.array<10xi32>>
! CHECK:           fir.call @_QFPd(%[[VAL_17]]) fastmath<contract> : (!fir.ref<!fir.array<10xi32>>) -> ()
! CHECK:           %[[VAL_18:.*]] = fir.embox %[[VAL_6]]#0(%[[VAL_4]]) : (!fir.ref<!fir.array<10xi32>, volatile>, !fir.shape<1>) -> !fir.box<!fir.array<10xi32>>
! CHECK:           %[[VAL_19:.*]] = fir.convert %[[VAL_18]] : (!fir.box<!fir.array<10xi32>>) -> !fir.box<!fir.array<?xi32>>
! CHECK:           fir.call @_QFPe(%[[VAL_19]]) fastmath<contract> : (!fir.box<!fir.array<?xi32>>) -> ()
! CHECK:           fir.call @_QFPf(%[[VAL_17]]) fastmath<contract> : (!fir.ref<!fir.array<10xi32>>) -> ()
! CHECK:           %[[VAL_20:.*]] = fir.convert %[[VAL_12]]#0 : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>, volatile>
! CHECK:           fir.call @_QFPg(%[[VAL_20]]) fastmath<contract> : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>, volatile>) -> ()
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func private @_QFPd(
! CHECK-SAME:                             %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.ref<!fir.array<10xi32>> {fir.bindc_name = "arr"}) attributes {{.+}} {
! CHECK:           %[[VAL_1:.*]] = arith.constant 10 : index
! CHECK:           %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_3:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_4:.*]] = fir.volatile_cast %[[VAL_0]] : (!fir.ref<!fir.array<10xi32>>) -> !fir.ref<!fir.array<10xi32>, volatile>
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_4]](%[[VAL_3]]) dummy_scope %[[VAL_2]] {{.+}} : (!fir.ref<!fir.array<10xi32>, volatile>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<10xi32>, volatile>, !fir.ref<!fir.array<10xi32>, volatile>)
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func private @_QFPe(
! CHECK-SAME:                             %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "arr"}) attributes {{.+}} {
! CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_1]] {{.+}} : (!fir.box<!fir.array<?xi32>>, !fir.dscope) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func private @_QFPf(
! CHECK-SAME:                             %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.ref<!fir.array<10xi32>> {fir.bindc_name = "arr"}) attributes {{.+}} {
! CHECK:           %[[VAL_1:.*]] = arith.constant 10 : index
! CHECK:           %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_3:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_4:.*]] = fir.volatile_cast %[[VAL_0]] : (!fir.ref<!fir.array<10xi32>>) -> !fir.ref<!fir.array<10xi32>, volatile>
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_4]](%[[VAL_3]]) dummy_scope %[[VAL_2]] {{.+}} : (!fir.ref<!fir.array<10xi32>, volatile>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<10xi32>, volatile>, !fir.ref<!fir.array<10xi32>, volatile>)
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func private @_QFPg(
! CHECK-SAME:                             %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>, volatile> {fir.bindc_name = "arr"}) attributes {{.+}} {
! CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_2:.*]] = fir.volatile_cast %[[VAL_0]] : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>, volatile>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]] dummy_scope %[[VAL_1]] {{.+}} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>)
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   fir.global internal @_QFEarr : !fir.array<10xi32> {
! CHECK:           %[[VAL_0:.*]] = fir.zero_bits !fir.array<10xi32>
! CHECK:           fir.has_value %[[VAL_0]] : !fir.array<10xi32>
! CHECK:         }

! CHECK-LABEL:   fir.global internal @_QFEptr : !fir.box<!fir.ptr<!fir.array<?xi32>>> {
! CHECK:           %[[VAL_0:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_1:.*]] = fir.zero_bits !fir.ptr<!fir.array<?xi32>>
! CHECK:           %[[VAL_2:.*]] = fir.shape %[[VAL_0]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_3:.*]] = fir.embox %[[VAL_1]](%[[VAL_2]]) : (!fir.ptr<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>>
! CHECK:           fir.has_value %[[VAL_3]] : !fir.box<!fir.ptr<!fir.array<?xi32>>>
! CHECK:         }

! CHECK-LABEL:   fir.global internal @_QFEtgt target : !fir.array<10xi32> {
! CHECK:           %[[VAL_0:.*]] = fir.zero_bits !fir.array<10xi32>
! CHECK:           fir.has_value %[[VAL_0]] : !fir.array<10xi32>
! CHECK:         }

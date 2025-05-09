! RUN: bbc --strict-fir-volatile-verifier %s -o - | FileCheck %s

program p
  integer,volatile::i,arr(10)
  integer,volatile,target::tgt(10)
  integer,volatile,pointer,dimension(:)::ptr
  i = loc(i)
  ptr => tgt
  i=0
  arr=1
  call host_assoc
contains
  subroutine host_assoc
    ptr => tgt
    i=0
    arr=1
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
! CHECK:           %[[VAL_16:.*]] = fir.alloca tuple<!fir.ref<i32>>
! CHECK:           %[[VAL_17:.*]] = fir.coordinate_of %[[VAL_16]], %[[VAL_1]] : (!fir.ref<tuple<!fir.ref<i32>>>, i32) -> !fir.llvm_ptr<!fir.ref<i32>>
! CHECK:           %[[VAL_18:.*]] = fir.volatile_cast %[[VAL_9]]#0 : (!fir.ref<i32, volatile>) -> !fir.ref<i32>
! CHECK:           fir.store %[[VAL_18]] to %[[VAL_17]] : !fir.llvm_ptr<!fir.ref<i32>>
! CHECK:           %[[VAL_19:.*]] = fir.embox %[[VAL_15]]#0(%[[VAL_4]]) : (!fir.ref<!fir.array<10xi32>, volatile>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>
! CHECK:           fir.store %[[VAL_19]] to %[[VAL_12]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>
! CHECK:           hlfir.assign %[[VAL_1]] to %[[VAL_9]]#0 : i32, !fir.ref<i32, volatile>
! CHECK:           hlfir.assign %[[VAL_0]] to %[[VAL_6]]#0 : i32, !fir.ref<!fir.array<10xi32>, volatile>
! CHECK:           fir.call @_QFPhost_assoc(%[[VAL_16]]) fastmath<contract> : (!fir.ref<tuple<!fir.ref<i32>>>) -> ()
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func private @_QFPhost_assoc(
! CHECK-SAME:                                      %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.ref<tuple<!fir.ref<i32>>> {fir.host_assoc}) attributes {{.+}} {
! CHECK:           %[[VAL_1:.*]] = arith.constant 1 : i32
! CHECK:           %[[VAL_2:.*]] = arith.constant 0 : i32
! CHECK:           %[[VAL_3:.*]] = arith.constant 10 : index
! CHECK:           %[[VAL_4:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_5:.*]] = fir.address_of(@_QFEarr) : !fir.ref<!fir.array<10xi32>>
! CHECK:           %[[VAL_6:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_7:.*]] = fir.volatile_cast %[[VAL_5]] : (!fir.ref<!fir.array<10xi32>>) -> !fir.ref<!fir.array<10xi32>, volatile>
! CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_7]](%[[VAL_6]]) {{.+}} : (!fir.ref<!fir.array<10xi32>, volatile>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>, volatile>, !fir.ref<!fir.array<10xi32>, volatile>)
! CHECK:           %[[VAL_9:.*]] = fir.address_of(@_QFEptr) : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:           %[[VAL_10:.*]] = fir.volatile_cast %[[VAL_9]] : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>
! CHECK:           %[[VAL_11:.*]]:2 = hlfir.declare %[[VAL_10]] {{.+}} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>)
! CHECK:           %[[VAL_12:.*]] = fir.address_of(@_QFEtgt) : !fir.ref<!fir.array<10xi32>>
! CHECK:           %[[VAL_13:.*]] = fir.volatile_cast %[[VAL_12]] : (!fir.ref<!fir.array<10xi32>>) -> !fir.ref<!fir.array<10xi32>, volatile>
! CHECK:           %[[VAL_14:.*]]:2 = hlfir.declare %[[VAL_13]](%[[VAL_6]]) {{.+}} : (!fir.ref<!fir.array<10xi32>, volatile>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>, volatile>, !fir.ref<!fir.array<10xi32>, volatile>)
! CHECK:           %[[VAL_15:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_2]] : (!fir.ref<tuple<!fir.ref<i32>>>, i32) -> !fir.llvm_ptr<!fir.ref<i32>>
! CHECK:           %[[VAL_16:.*]] = fir.load %[[VAL_15]] : !fir.llvm_ptr<!fir.ref<i32>>
! CHECK:           %[[VAL_17:.*]] = fir.volatile_cast %[[VAL_16]] : (!fir.ref<i32>) -> !fir.ref<i32, volatile>
! CHECK:           %[[VAL_18:.*]]:2 = hlfir.declare %[[VAL_17]] {{.+}} : (!fir.ref<i32, volatile>) -> (!fir.ref<i32, volatile>, !fir.ref<i32, volatile>)
! CHECK:           %[[VAL_19:.*]] = fir.embox %[[VAL_14]]#0(%[[VAL_6]]) : (!fir.ref<!fir.array<10xi32>, volatile>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>
! CHECK:           fir.store %[[VAL_19]] to %[[VAL_11]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>
! CHECK:           hlfir.assign %[[VAL_2]] to %[[VAL_18]]#0 : i32, !fir.ref<i32, volatile>
! CHECK:           hlfir.assign %[[VAL_1]] to %[[VAL_8]]#0 : i32, !fir.ref<!fir.array<10xi32>, volatile>
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

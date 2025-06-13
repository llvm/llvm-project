! RUN: bbc --strict-fir-volatile-verifier %s -o - | FileCheck %s

program p
  integer,volatile::i,arr(10)
  call not_declared_volatile_in_this_scope(i)
  call not_declared_volatile_in_this_scope(arr)
  call declared_volatile_in_this_scope(arr,10)
  print*,arr,i
contains
  elemental subroutine not_declared_volatile_in_this_scope(v)
    integer,intent(inout)::v
    v=1
  end subroutine
  subroutine declared_volatile_in_this_scope(v,n)
    integer,intent(in)::n
    integer,volatile,intent(inout)::v(n)
    v=1
  end subroutine
end program


! CHECK-LABEL:   func.func @_QQmain() attributes {{.+}} {
! CHECK:           %[[VAL_0:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_1:.*]] = arith.constant 8 : i32
! CHECK:           %[[VAL_2:.*]] = arith.constant 6 : i32
! CHECK:           %[[VAL_3:.*]] = arith.constant 10 : i32
! CHECK:           %[[VAL_4:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_5:.*]] = arith.constant 10 : index
! CHECK:           %[[VAL_6:.*]] = fir.address_of(@_QFEarr) : !fir.ref<!fir.array<10xi32>>
! CHECK:           %[[VAL_7:.*]] = fir.shape %[[VAL_5]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_8:.*]] = fir.volatile_cast %[[VAL_6]] : (!fir.ref<!fir.array<10xi32>>) -> !fir.ref<!fir.array<10xi32>, volatile>
! CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_8]](%[[VAL_7]]) {{.+}} : (!fir.ref<!fir.array<10xi32>, volatile>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>, volatile>, !fir.ref<!fir.array<10xi32>, volatile>)
! CHECK:           %[[VAL_10:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFEi"}
! CHECK:           %[[VAL_11:.*]] = fir.volatile_cast %[[VAL_10]] : (!fir.ref<i32>) -> !fir.ref<i32, volatile>
! CHECK:           %[[VAL_12:.*]]:2 = hlfir.declare %[[VAL_11]] {{.+}} : (!fir.ref<i32, volatile>) -> (!fir.ref<i32, volatile>, !fir.ref<i32, volatile>)
! CHECK:           %[[VAL_13:.*]] = fir.volatile_cast %[[VAL_12]]#0 : (!fir.ref<i32, volatile>) -> !fir.ref<i32>
! CHECK:           fir.call @_QFPnot_declared_volatile_in_this_scope(%[[VAL_13]]) proc_attrs<elemental, pure> fastmath<contract> : (!fir.ref<i32>) -> ()
! CHECK:           cf.br ^bb1(%[[VAL_4]], %[[VAL_5]] : index, index)
! CHECK:         ^bb1(%[[VAL_14:.*]]: index, %[[VAL_15:.*]]: index):
! CHECK:           %[[VAL_16:.*]] = arith.cmpi sgt, %[[VAL_15]], %[[VAL_0]] : index
! CHECK:           cf.cond_br %[[VAL_16]], ^bb2, ^bb3
! CHECK:         ^bb2:
! CHECK:           %[[VAL_17:.*]] = hlfir.designate %[[VAL_9]]#0 (%[[VAL_14]])  : (!fir.ref<!fir.array<10xi32>, volatile>, index) -> !fir.ref<i32, volatile>
! CHECK:           %[[VAL_18:.*]] = fir.volatile_cast %[[VAL_17]] : (!fir.ref<i32, volatile>) -> !fir.ref<i32>
! CHECK:           fir.call @_QFPnot_declared_volatile_in_this_scope(%[[VAL_18]]) proc_attrs<elemental, pure> fastmath<contract> : (!fir.ref<i32>) -> ()
! CHECK:           %[[VAL_19:.*]] = arith.addi %[[VAL_14]], %[[VAL_4]] overflow<nsw> : index
! CHECK:           %[[VAL_20:.*]] = arith.subi %[[VAL_15]], %[[VAL_4]] : index
! CHECK:           cf.br ^bb1(%[[VAL_19]], %[[VAL_20]] : index, index)
! CHECK:         ^bb3:
! CHECK:           %[[VAL_21:.*]] = fir.volatile_cast %[[VAL_9]]#0 : (!fir.ref<!fir.array<10xi32>, volatile>) -> !fir.ref<!fir.array<10xi32>>
! CHECK:           %[[VAL_22:.*]] = fir.convert %[[VAL_21]] : (!fir.ref<!fir.array<10xi32>>) -> !fir.ref<!fir.array<?xi32>>
! CHECK:           %[[VAL_23:.*]]:3 = hlfir.associate %[[VAL_3]] {adapt.valuebyref} : (i32) -> (!fir.ref<i32>, !fir.ref<i32>, i1)
! CHECK:           fir.call @_QFPdeclared_volatile_in_this_scope(%[[VAL_22]], %[[VAL_23]]#0) fastmath<contract> : (!fir.ref<!fir.array<?xi32>>, !fir.ref<i32>) -> ()
! CHECK:           hlfir.end_associate %[[VAL_23]]#1, %[[VAL_23]]#2 : !fir.ref<i32>, i1
! CHECK:           %[[VAL_24:.*]] = fir.address_of
! CHECK:           %[[VAL_25:.*]] = fir.convert %[[VAL_24]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_26:.*]] = fir.call @_FortranAioBeginExternalListOutput(%[[VAL_2]], %[[VAL_25]], %[[VAL_1]]) fastmath<contract> {{.+}} : (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:           %[[VAL_27:.*]] = fir.embox %[[VAL_9]]#0(%[[VAL_7]]) : (!fir.ref<!fir.array<10xi32>, volatile>, !fir.shape<1>) -> !fir.box<!fir.array<10xi32>, volatile>
! CHECK:           %[[VAL_28:.*]] = fir.volatile_cast %[[VAL_27]] : (!fir.box<!fir.array<10xi32>, volatile>) -> !fir.box<!fir.array<10xi32>>
! CHECK:           %[[VAL_29:.*]] = fir.convert %[[VAL_28]] : (!fir.box<!fir.array<10xi32>>) -> !fir.box<none>
! CHECK:           %[[VAL_30:.*]] = fir.call @_FortranAioOutputDescriptor(%[[VAL_26]], %[[VAL_29]]) fastmath<contract> {llvm.nocallback, llvm.nosync} : (!fir.ref<i8>, !fir.box<none>) -> i1
! CHECK:           %[[VAL_31:.*]] = fir.load %[[VAL_12]]#0 : !fir.ref<i32, volatile>
! CHECK:           %[[VAL_32:.*]] = fir.call @_FortranAioOutputInteger32(%[[VAL_26]], %[[VAL_31]]) fastmath<contract> {{.+}} : (!fir.ref<i8>, i32) -> i1
! CHECK:           %[[VAL_33:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_26]]) fastmath<contract> {{.+}} : (!fir.ref<i8>) -> i32
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func private @_QFPnot_declared_volatile_in_this_scope(
! CHECK-SAME:                                                               %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.ref<i32> {fir.bindc_name = "v"}) attributes {{.+}} {
! CHECK:           %[[VAL_1:.*]] = arith.constant 1 : i32
! CHECK:           %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_2]] {{.+}} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           hlfir.assign %[[VAL_1]] to %[[VAL_3]]#0 : i32, !fir.ref<i32>
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func private @_QFPdeclared_volatile_in_this_scope(
! CHECK-SAME:                                                           %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.ref<!fir.array<?xi32>> {fir.bindc_name = "v"},
! CHECK-SAME:                                                           %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.ref<i32> {fir.bindc_name = "n"}) attributes {{.+}} {
! CHECK:           %[[VAL_2:.*]] = arith.constant 1 : i32
! CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_4:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_1]] dummy_scope %[[VAL_4]] {{.+}} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_6:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<i32>
! CHECK:           %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (i32) -> index
! CHECK:           %[[VAL_8:.*]] = arith.cmpi sgt, %[[VAL_7]], %[[VAL_3]] : index
! CHECK:           %[[VAL_9:.*]] = arith.select %[[VAL_8]], %[[VAL_7]], %[[VAL_3]] : index
! CHECK:           %[[VAL_10:.*]] = fir.shape %[[VAL_9]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_11:.*]] = fir.volatile_cast %[[VAL_0]] : (!fir.ref<!fir.array<?xi32>>) -> !fir.ref<!fir.array<?xi32>, volatile>
! CHECK:           %[[VAL_12:.*]]:2 = hlfir.declare %[[VAL_11]](%[[VAL_10]]) dummy_scope %[[VAL_4]] {{.+}} : (!fir.ref<!fir.array<?xi32>, volatile>, !fir.shape<1>, !fir.dscope) -> (!fir.box<!fir.array<?xi32>, volatile>, !fir.ref<!fir.array<?xi32>, volatile>)
! CHECK:           hlfir.assign %[[VAL_2]] to %[[VAL_12]]#0 : i32, !fir.box<!fir.array<?xi32>, volatile>
! CHECK:           return
! CHECK:         }

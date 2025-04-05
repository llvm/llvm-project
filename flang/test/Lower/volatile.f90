! RUN: bbc %s -o - | FileCheck %s

program p
  integer,volatile::i,arr(10)
  i=0
  arr=1
  ! casting from volatile ref to non-volatile ref should be okay here
  call not_declared_volatile_in_this_scope(i)
  call not_declared_volatile_in_this_scope(arr)
  call declared_volatile_in_this_scope(arr,10)
  print*,arr,i,a(),b(),c()
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
  function a()
    integer,volatile::a
    a=1
  end function
  function b() result(r)
    integer,volatile::r
    r=2
  end function
  function c() result(r)
    volatile::r
    r=3
  end function
end program

! CHECK-LABEL:   func.func @_QQmain
! CHECK:           %[[VAL_0:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_1:.*]] = arith.constant 11 : i32
! CHECK:           %[[VAL_2:.*]] = arith.constant 6 : i32
! CHECK:           %[[VAL_3:.*]] = arith.constant 10 : i32
! CHECK:           %[[VAL_4:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_5:.*]] = arith.constant 1 : i32
! CHECK:           %[[VAL_6:.*]] = arith.constant 0 : i32
! CHECK:           %[[VAL_7:.*]] = arith.constant 10 : index
! CHECK:           %[[VAL_8:.*]] = fir.address_of(@_QFEarr) : !fir.ref<!fir.array<10xi32>>
! CHECK:           %[[VAL_9:.*]] = fir.shape %[[VAL_7]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_10:.*]] = fir.volatile_cast %[[VAL_8]] : (!fir.ref<!fir.array<10xi32>>) -> !fir.ref<!fir.array<10xi32>, volatile>
! CHECK:           %[[VAL_11:.*]]:2 = hlfir.declare %[[VAL_10]](%[[VAL_9]]) {{.*}} : (!fir.ref<!fir.array<10xi32>, volatile>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>, volatile>, !fir.ref<!fir.array<10xi32>, volatile>)
! CHECK:           %[[VAL_12:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFEi"}
! CHECK:           %[[VAL_13:.*]] = fir.volatile_cast %[[VAL_12]] : (!fir.ref<i32>) -> !fir.ref<i32, volatile>
! CHECK:           %[[VAL_14:.*]]:2 = hlfir.declare %[[VAL_13]] {{.*}} : (!fir.ref<i32, volatile>) -> (!fir.ref<i32, volatile>, !fir.ref<i32, volatile>)
! CHECK:           hlfir.assign %[[VAL_6]] to %[[VAL_14]]#0 : i32, !fir.ref<i32, volatile>
! CHECK:           hlfir.assign %[[VAL_5]] to %[[VAL_11]]#0 : i32, !fir.ref<!fir.array<10xi32>, volatile>
! CHECK:           %[[VAL_15:.*]] = fir.volatile_cast %[[VAL_14]]#0 : (!fir.ref<i32, volatile>) -> !fir.ref<i32>
! CHECK:           fir.call @_QFPnot_declared_volatile_in_this_scope(%[[VAL_15]]) proc_attrs<elemental, pure> fastmath<contract> : (!fir.ref<i32>) -> ()
! CHECK:           cf.br ^bb1(%[[VAL_4]], %[[VAL_7]] : index, index)
! CHECK:         ^bb1(%[[VAL_16:.*]]: index, %[[VAL_17:.*]]: index):
! CHECK:           %[[VAL_18:.*]] = arith.cmpi sgt, %[[VAL_17]], %[[VAL_0]] : index
! CHECK:           cf.cond_br %[[VAL_18]], ^bb2, ^bb3
! CHECK:         ^bb2:
! CHECK:           %[[VAL_19:.*]] = hlfir.designate %[[VAL_11]]#0 (%[[VAL_16]])  : (!fir.ref<!fir.array<10xi32>, volatile>, index) -> !fir.ref<i32, volatile>
! CHECK:           %[[VAL_20:.*]] = fir.volatile_cast %[[VAL_19]] : (!fir.ref<i32, volatile>) -> !fir.ref<i32>
! CHECK:           fir.call @_QFPnot_declared_volatile_in_this_scope(%[[VAL_20]]) proc_attrs<elemental, pure> fastmath<contract> : (!fir.ref<i32>) -> ()
! CHECK:           %[[VAL_21:.*]] = arith.addi %[[VAL_16]], %[[VAL_4]] overflow<nsw> : index
! CHECK:           %[[VAL_22:.*]] = arith.subi %[[VAL_17]], %[[VAL_4]] : index
! CHECK:           cf.br ^bb1(%[[VAL_21]], %[[VAL_22]] : index, index)
! CHECK:         ^bb3:
! CHECK:           %[[VAL_23:.*]] = fir.volatile_cast %[[VAL_11]]#0 : (!fir.ref<!fir.array<10xi32>, volatile>) -> !fir.ref<!fir.array<10xi32>>
! CHECK:           %[[VAL_24:.*]] = fir.convert %[[VAL_23]] : (!fir.ref<!fir.array<10xi32>>) -> !fir.ref<!fir.array<?xi32>>
! CHECK:           %[[VAL_25:.*]]:3 = hlfir.associate %[[VAL_3]] {adapt.valuebyref} : (i32) -> (!fir.ref<i32>, !fir.ref<i32>, i1)
! CHECK:           fir.call @_QFPdeclared_volatile_in_this_scope(%[[VAL_24]], %[[VAL_25]]#0) fastmath<contract> : (!fir.ref<!fir.array<?xi32>>, !fir.ref<i32>) -> ()
! CHECK:           hlfir.end_associate %[[VAL_25]]#1, %[[VAL_25]]#2 : !fir.ref<i32>, i1
! CHECK:           %[[VAL_26:.*]] = fir.address_of
! CHECK:           %[[VAL_27:.*]] = fir.convert %[[VAL_26]] : (!fir.ref<!fir.char<1,{{.+}}>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_28:.*]] = fir.call @_FortranAioBeginExternalListOutput(%[[VAL_2]], %[[VAL_27]], %[[VAL_1]]) fastmath<contract>
! CHECK:           %[[VAL_29:.*]] = fir.embox %[[VAL_11]]#0(%[[VAL_9]]) : (!fir.ref<!fir.array<10xi32>, volatile>, !fir.shape<1>) -> !fir.box<!fir.array<10xi32>, volatile>
! CHECK:           %[[VAL_30:.*]] = fir.volatile_cast %[[VAL_29]] : (!fir.box<!fir.array<10xi32>, volatile>) -> !fir.box<!fir.array<10xi32>>
! CHECK:           %[[VAL_31:.*]] = fir.convert %[[VAL_30]] : (!fir.box<!fir.array<10xi32>>) -> !fir.box<none>
! CHECK:           %[[VAL_32:.*]] = fir.call @_FortranAioOutputDescriptor(%[[VAL_28]], %[[VAL_31]])
! CHECK:           %[[VAL_33:.*]] = fir.load %[[VAL_14]]#0 : !fir.ref<i32, volatile>
! CHECK:           %[[VAL_34:.*]] = fir.call @_FortranAioOutputInteger32(%[[VAL_28]], %[[VAL_33]]) 
! CHECK:           %[[VAL_35:.*]] = fir.call @_QFPa() fastmath<contract> : () -> i32
! CHECK:           %[[VAL_36:.*]] = fir.call @_FortranAioOutputInteger32(%[[VAL_28]], %[[VAL_35]]) 
! CHECK:           %[[VAL_37:.*]] = fir.call @_QFPb() fastmath<contract> : () -> i32
! CHECK:           %[[VAL_38:.*]] = fir.call @_FortranAioOutputInteger32(%[[VAL_28]], %[[VAL_37]])
! CHECK:           %[[VAL_39:.*]] = fir.call @_QFPc() fastmath<contract> : () -> f32
! CHECK:           %[[VAL_40:.*]] = fir.call @_FortranAioOutputReal32(%[[VAL_28]], %[[VAL_39]])
! CHECK:           %[[VAL_41:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_28]])
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func private @_QFPnot_declared_volatile_in_this_scope(
! CHECK-SAME:                                                               %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.ref<i32> {fir.bindc_name = "v"}) attributes {{.+}} {
! CHECK:           %[[VAL_1:.*]] = arith.constant 1 : i32
! CHECK:           %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_2]] {{.*}} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           hlfir.assign %[[VAL_1]] to %[[VAL_3]]#0 : i32, !fir.ref<i32>
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func private @_QFPdeclared_volatile_in_this_scope(
! CHECK-SAME:                                                           %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.ref<!fir.array<?xi32>> {fir.bindc_name = "v", fir.volatile},
! CHECK-SAME:                                                           %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.ref<i32> {fir.bindc_name = "n"}) attributes {{.+}} {
! CHECK:           %[[VAL_2:.*]] = arith.constant 1 : i32
! CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_4:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_1]] dummy_scope %[[VAL_4]] {{.*}} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_6:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<i32>
! CHECK:           %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (i32) -> index
! CHECK:           %[[VAL_8:.*]] = arith.cmpi sgt, %[[VAL_7]], %[[VAL_3]] : index
! CHECK:           %[[VAL_9:.*]] = arith.select %[[VAL_8]], %[[VAL_7]], %[[VAL_3]] : index
! CHECK:           %[[VAL_10:.*]] = fir.shape %[[VAL_9]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_11:.*]] = fir.volatile_cast %[[VAL_0]] : (!fir.ref<!fir.array<?xi32>>) -> !fir.ref<!fir.array<?xi32>, volatile>
! CHECK:           %[[VAL_12:.*]]:2 = hlfir.declare %[[VAL_11]](%[[VAL_10]]) dummy_scope %[[VAL_4]] {{.*}} : (!fir.ref<!fir.array<?xi32>, volatile>, !fir.shape<1>, !fir.dscope) -> (!fir.box<!fir.array<?xi32>, volatile>, !fir.ref<!fir.array<?xi32>, volatile>)
! CHECK:           hlfir.assign %[[VAL_2]] to %[[VAL_12]]#0 : i32, !fir.box<!fir.array<?xi32>, volatile>
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func private @_QFPa() -> i32 attributes {{.+}} {
! CHECK:           %[[VAL_0:.*]] = arith.constant 1 : i32
! CHECK:           %[[VAL_1:.*]] = fir.alloca i32 {bindc_name = "a", uniq_name = "_QFFaEa"}
! CHECK:           %[[VAL_2:.*]] = fir.volatile_cast %[[VAL_1]] : (!fir.ref<i32>) -> !fir.ref<i32, volatile>
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]] {{.*}} : (!fir.ref<i32, volatile>) -> (!fir.ref<i32, volatile>, !fir.ref<i32, volatile>)
! CHECK:           hlfir.assign %[[VAL_0]] to %[[VAL_3]]#0 : i32, !fir.ref<i32, volatile>
! CHECK:           %[[VAL_4:.*]] = fir.volatile_cast %[[VAL_3]]#0 : (!fir.ref<i32, volatile>) -> !fir.ref<i32>
! CHECK:           %[[VAL_5:.*]] = fir.load %[[VAL_4]] : !fir.ref<i32>
! CHECK:           return %[[VAL_5]] : i32
! CHECK:         }

! CHECK-LABEL:   func.func private @_QFPb() -> i32 attributes {{.+}} {
! CHECK:           %[[VAL_0:.*]] = arith.constant 2 : i32
! CHECK:           %[[VAL_1:.*]] = fir.alloca i32 {bindc_name = "r", uniq_name = "_QFFbEr"}
! CHECK:           %[[VAL_2:.*]] = fir.volatile_cast %[[VAL_1]] : (!fir.ref<i32>) -> !fir.ref<i32, volatile>
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]] {{.*}} : (!fir.ref<i32, volatile>) -> (!fir.ref<i32, volatile>, !fir.ref<i32, volatile>)
! CHECK:           hlfir.assign %[[VAL_0]] to %[[VAL_3]]#0 : i32, !fir.ref<i32, volatile>
! CHECK:           %[[VAL_4:.*]] = fir.volatile_cast %[[VAL_3]]#0 : (!fir.ref<i32, volatile>) -> !fir.ref<i32>
! CHECK:           %[[VAL_5:.*]] = fir.load %[[VAL_4]] : !fir.ref<i32>
! CHECK:           return %[[VAL_5]] : i32
! CHECK:         }

! CHECK-LABEL:   func.func private @_QFPc() -> f32 attributes {{.+}} {
! CHECK:           %[[VAL_0:.*]] = arith.constant 3.000000e+00 : f32
! CHECK:           %[[VAL_1:.*]] = fir.alloca f32 {bindc_name = "r", uniq_name = "_QFFcEr"}
! CHECK:           %[[VAL_2:.*]] = fir.volatile_cast %[[VAL_1]] : (!fir.ref<f32>) -> !fir.ref<f32, volatile>
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]] {{.*}} : (!fir.ref<f32, volatile>) -> (!fir.ref<f32, volatile>, !fir.ref<f32, volatile>)
! CHECK:           hlfir.assign %[[VAL_0]] to %[[VAL_3]]#0 : f32, !fir.ref<f32, volatile>
! CHECK:           %[[VAL_4:.*]] = fir.volatile_cast %[[VAL_3]]#0 : (!fir.ref<f32, volatile>) -> !fir.ref<f32>
! CHECK:           %[[VAL_5:.*]] = fir.load %[[VAL_4]] : !fir.ref<f32>
! CHECK:           return %[[VAL_5]] : f32
! CHECK:         }

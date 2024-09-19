! RUN: bbc -emit-fir -o - %s | FileCheck %s

! CHECK-LABEL: c.func @_QQmain
program p
  ! CHECK:     %[[V_1:[0-9]+]] = fir.alloca !fir.box<!fir.ptr<!fir.array<4x!fir.char<1,3>>>>
  ! CHECK:     %[[V_2:[0-9]+]] = fir.alloca !fir.box<!fir.ptr<i32>>
  ! CHECK:     %[[V_3:[0-9]+]] = fir.alloca !fir.box<!fir.ptr<!fir.array<4x!fir.char<1,3>>>>
  ! CHECK:     %[[V_4:[0-9]+]] = fir.alloca !fir.box<!fir.ptr<i32>>
  ! CHECK:     %[[V_5:[0-9]+]] = fir.alloca !fir.array<4x!fir.char<1,3>> {bindc_name = "ccc", uniq_name = "_QFEccc"}
  ! CHECK:     %[[V_6:[0-9]+]] = fir.shape %c4{{.*}} : (index) -> !fir.shape<1>
  ! CHECK:     %[[V_7:[0-9]+]] = fir.declare %[[V_5]](%[[V_6]]) typeparams %c3{{.*}} {uniq_name = "_QFEccc"} : (!fir.ref<!fir.array<4x!fir.char<1,3>>>, !fir.shape<1>, index) -> !fir.ref<!fir.array<4x!fir.char<1,3>>>
  ! CHECK:     %[[V_8:[0-9]+]] = fir.alloca i32 {bindc_name = "jjj", uniq_name = "_QFEjjj"}
  ! CHECK:     %[[V_9:[0-9]+]] = fir.declare %[[V_8]] {uniq_name = "_QFEjjj"} : (!fir.ref<i32>) -> !fir.ref<i32>
  ! CHECK:     fir.store %c17{{.*}} to %[[V_9]] : !fir.ref<i32>
  character*3 ccc(4)
  namelist /nnn/ jjj, ccc
  jjj = 17
  ccc = ["aa ", "bb ", "cc ", "dd "]

  ! CHECK:     %[[V_23:[0-9]+]] = fir.call @_FortranAioBeginExternalListOutput
  ! CHECK:     %[[V_24:[0-9]+]] = fir.alloca !fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>
  ! CHECK:     %[[V_25:[0-9]+]] = fir.undefined !fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>
  ! CHECK:     %[[V_26:[0-9]+]] = fir.address_of(@_QQclX6A6A6A00) : !fir.ref<!fir.char<1,4>>
  ! CHECK:     %[[V_27:[0-9]+]] = fir.convert %[[V_26]] : (!fir.ref<!fir.char<1,4>>) -> !fir.ref<i8>
  ! CHECK:     %[[V_28:[0-9]+]] = fir.insert_value %[[V_25]], %[[V_27]], [0 : index, 0 : index] : (!fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>, !fir.ref<i8>) -> !fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>
  ! CHECK:     %[[V_29:[0-9]+]] = fir.embox %[[V_9]] : (!fir.ref<i32>) -> !fir.box<!fir.ptr<i32>>
  ! CHECK:     fir.store %[[V_29]] to %[[V_4]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
  ! CHECK:     %[[V_30:[0-9]+]] = fir.convert %[[V_4]] : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK:     %[[V_31:[0-9]+]] = fir.insert_value %[[V_28]], %[[V_30]], [0 : index, 1 : index] : (!fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>, !fir.ref<!fir.box<none>>) -> !fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>
  ! CHECK:     %[[V_32:[0-9]+]] = fir.address_of(@_QQclX63636300) : !fir.ref<!fir.char<1,4>>
  ! CHECK:     %[[V_33:[0-9]+]] = fir.convert %[[V_32]] : (!fir.ref<!fir.char<1,4>>) -> !fir.ref<i8>
  ! CHECK:     %[[V_34:[0-9]+]] = fir.insert_value %[[V_31]], %[[V_33]], [1 : index, 0 : index] : (!fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>, !fir.ref<i8>) -> !fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>
  ! CHECK:     %[[V_35:[0-9]+]] = fir.embox %[[V_7]](%[[V_6]]) : (!fir.ref<!fir.array<4x!fir.char<1,3>>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<4x!fir.char<1,3>>>>
  ! CHECK:     fir.store %[[V_35]] to %[[V_3]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<4x!fir.char<1,3>>>>>
  ! CHECK:     %[[V_36:[0-9]+]] = fir.convert %[[V_3]] : (!fir.ref<!fir.box<!fir.ptr<!fir.array<4x!fir.char<1,3>>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK:     %[[V_37:[0-9]+]] = fir.insert_value %[[V_34]], %[[V_36]], [1 : index, 1 : index] : (!fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>, !fir.ref<!fir.box<none>>) -> !fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>
  ! CHECK:     fir.store %[[V_37]] to %[[V_24]] : !fir.ref<!fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>
  ! CHECK:     %[[V_38:[0-9]+]] = fir.alloca tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>, !fir.ref<none>>
  ! CHECK:     %[[V_39:[0-9]+]] = fir.undefined tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>, !fir.ref<none>>
  ! CHECK:     %[[V_40:[0-9]+]] = fir.address_of(@_QQclX6E6E6E00) : !fir.ref<!fir.char<1,4>>
  ! CHECK:     %[[V_41:[0-9]+]] = fir.convert %[[V_40]] : (!fir.ref<!fir.char<1,4>>) -> !fir.ref<i8>
  ! CHECK:     %[[V_42:[0-9]+]] = fir.insert_value %[[V_39]], %[[V_41]], [0 : index] : (tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>, !fir.ref<none>>, !fir.ref<i8>) -> tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>, !fir.ref<none>>
  ! CHECK:     %[[V_43:[0-9]+]] = fir.insert_value %[[V_42]], %c2{{.*}}, [1 : index] : (tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>, !fir.ref<none>>, i64) -> tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>, !fir.ref<none>>
  ! CHECK:     %[[V_44:[0-9]+]] = fir.insert_value %[[V_43]], %[[V_24]], [2 : index] : (tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>, !fir.ref<none>>, !fir.ref<!fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>) -> tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>, !fir.ref<none>>
  ! CHECK:     %[[V_45:[0-9]+]] = fir.address_of(@default.nonTbpDefinedIoTable) : !fir.ref<tuple<i64, !fir.ref<!fir.array<0xtuple<!fir.ref<none>, !fir.ref<none>, i32, i1>>>, i1>>
  ! CHECK:     %[[V_46:[0-9]+]] = fir.convert %[[V_45]] : (!fir.ref<tuple<i64, !fir.ref<!fir.array<0xtuple<!fir.ref<none>, !fir.ref<none>, i32, i1>>>, i1>>) -> !fir.ref<none>
  ! CHECK:     %[[V_47:[0-9]+]] = fir.insert_value %[[V_44]], %[[V_46]], [3 : index] : (tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>, !fir.ref<none>>, !fir.ref<none>) -> tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>, !fir.ref<none>>
  ! CHECK:     fir.store %[[V_47]] to %[[V_38]] : !fir.ref<tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>, !fir.ref<none>>>
  ! CHECK:     %[[V_48:[0-9]+]] = fir.convert %[[V_38]] : (!fir.ref<tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>, !fir.ref<none>>>) -> !fir.ref<tuple<>>
  ! CHECK:     %[[V_49:[0-9]+]] = fir.call @_FortranAioOutputNamelist(%[[V_23]], %[[V_48]]) fastmath<contract> : (!fir.ref<i8>, !fir.ref<tuple<>>) -> i1
  ! CHECK:     %[[V_50:[0-9]+]] = fir.call @_FortranAioEndIoStatement(%[[V_23]]) fastmath<contract> : (!fir.ref<i8>) -> i32
  write(*, nnn)
  jjj = 27
  ccc(4) = "zz "
  ! CHECK:     %[[V_58:[0-9]+]] = fir.call @_FortranAioBeginExternalListOutput
  ! CHECK:     %[[V_59:[0-9]+]] = fir.alloca !fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>
  ! CHECK:     fir.store %[[V_29]] to %[[V_2]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
  ! CHECK:     %[[V_60:[0-9]+]] = fir.convert %[[V_2]] : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK:     %[[V_61:[0-9]+]] = fir.insert_value %[[V_28]], %[[V_60]], [0 : index, 1 : index] : (!fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>, !fir.ref<!fir.box<none>>) -> !fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>
  ! CHECK:     %[[V_62:[0-9]+]] = fir.insert_value %[[V_61]], %[[V_33]], [1 : index, 0 : index] : (!fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>, !fir.ref<i8>) -> !fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>
  ! CHECK:     fir.store %[[V_35]] to %[[V_1]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<4x!fir.char<1,3>>>>>
  ! CHECK:     %[[V_63:[0-9]+]] = fir.convert %[[V_1]] : (!fir.ref<!fir.box<!fir.ptr<!fir.array<4x!fir.char<1,3>>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK:     %[[V_64:[0-9]+]] = fir.insert_value %[[V_62]], %[[V_63]], [1 : index, 1 : index] : (!fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>, !fir.ref<!fir.box<none>>) -> !fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>
  ! CHECK:     fir.store %[[V_64]] to %[[V_59]] : !fir.ref<!fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>
  ! CHECK:     %[[V_65:[0-9]+]] = fir.alloca tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>, !fir.ref<none>>
  ! CHECK:     %[[V_66:[0-9]+]] = fir.insert_value %[[V_43]], %[[V_59]], [2 : index] : (tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>, !fir.ref<none>>, !fir.ref<!fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>) -> tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>, !fir.ref<none>>
  ! CHECK:     %[[V_67:[0-9]+]] = fir.insert_value %[[V_66]], %[[V_46]], [3 : index] : (tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>, !fir.ref<none>>, !fir.ref<none>) -> tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>, !fir.ref<none>>
  ! CHECK:     fir.store %[[V_67]] to %[[V_65]] : !fir.ref<tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>, !fir.ref<none>>>
  ! CHECK:     %[[V_68:[0-9]+]] = fir.convert %[[V_65]] : (!fir.ref<tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>, !fir.ref<none>>>) -> !fir.ref<tuple<>>
  ! CHECK:     %[[V_69:[0-9]+]] = fir.call @_FortranAioOutputNamelist(%[[V_58]], %[[V_68]]) fastmath<contract> : (!fir.ref<i8>, !fir.ref<tuple<>>) -> i1
  ! CHECK:     %[[V_70:[0-9]+]] = fir.call @_FortranAioEndIoStatement(%[[V_58]]) fastmath<contract> : (!fir.ref<i8>) -> i32
  write(*, nnn)

  call rename_sub
end

! CHECK-LABEL: c.func @_QPsss
subroutine sss
  ! CHECK:     %[[V_0:[0-9]+]] = fir.alloca !fir.box<!fir.ptr<!fir.array<3xi32>>>
  ! CHECK:     %[[V_1:[0-9]+]] = fir.alloca !fir.array<3xi32> {bindc_name = "xxx", uniq_name = "_QFsssExxx"}
  ! CHECK:     %[[V_2:[0-9]+]] = fir.shape_shift %c11{{.*}}, %c3{{.*}} : (index, index) -> !fir.shapeshift<1>
  ! CHECK:     %[[V_3:[0-9]+]] = fir.declare %[[V_1]](%[[V_2]]) {uniq_name = "_QFsssExxx"} : (!fir.ref<!fir.array<3xi32>>, !fir.shapeshift<1>) -> !fir.ref<!fir.array<3xi32>>
  integer xxx(11:13)

  ! CHECK:     %[[V_7:[0-9]+]] = fir.call @_FortranAioBeginExternalListInput
  ! CHECK:     %[[V_8:[0-9]+]] = fir.alloca !fir.array<1xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>
  ! CHECK:     %[[V_9:[0-9]+]] = fir.undefined !fir.array<1xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>
  ! CHECK:     %[[V_10:[0-9]+]] = fir.address_of(@_QQclX78787800) : !fir.ref<!fir.char<1,4>>
  ! CHECK:     %[[V_11:[0-9]+]] = fir.convert %[[V_10]] : (!fir.ref<!fir.char<1,4>>) -> !fir.ref<i8>
  ! CHECK:     %[[V_12:[0-9]+]] = fir.insert_value %[[V_9]], %[[V_11]], [0 : index, 0 : index] : (!fir.array<1xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>, !fir.ref<i8>) -> !fir.array<1xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>
  ! CHECK:     %[[V_13:[0-9]+]] = fir.embox %[[V_3]](%[[V_2]]) : (!fir.ref<!fir.array<3xi32>>, !fir.shapeshift<1>) -> !fir.box<!fir.ptr<!fir.array<3xi32>>>
  ! CHECK:     fir.store %[[V_13]] to %[[V_0]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<3xi32>>>>
  ! CHECK:     %[[V_14:[0-9]+]] = fir.convert %[[V_0]] : (!fir.ref<!fir.box<!fir.ptr<!fir.array<3xi32>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK:     %[[V_15:[0-9]+]] = fir.insert_value %[[V_12]], %[[V_14]], [0 : index, 1 : index] : (!fir.array<1xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>, !fir.ref<!fir.box<none>>) -> !fir.array<1xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>
  ! CHECK:     fir.store %[[V_15]] to %[[V_8]] : !fir.ref<!fir.array<1xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>
  ! CHECK:     %[[V_16:[0-9]+]] = fir.alloca tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<1xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>, !fir.ref<none>>
  ! CHECK:     %[[V_17:[0-9]+]] = fir.undefined tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<1xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>, !fir.ref<none>>
  ! CHECK:     %[[V_18:[0-9]+]] = fir.address_of(@_QQclX72727200) : !fir.ref<!fir.char<1,4>>
  ! CHECK:     %[[V_19:[0-9]+]] = fir.convert %[[V_18]] : (!fir.ref<!fir.char<1,4>>) -> !fir.ref<i8>
  ! CHECK:     %[[V_20:[0-9]+]] = fir.insert_value %[[V_17]], %[[V_19]], [0 : index] : (tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<1xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>, !fir.ref<none>>, !fir.ref<i8>) -> tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<1xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>, !fir.ref<none>>
  ! CHECK:     %[[V_21:[0-9]+]] = fir.insert_value %[[V_20]], %c1{{.*}}, [1 : index] : (tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<1xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>, !fir.ref<none>>, i64) -> tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<1xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>, !fir.ref<none>>
  ! CHECK:     %[[V_22:[0-9]+]] = fir.insert_value %[[V_21]], %[[V_8]], [2 : index] : (tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<1xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>, !fir.ref<none>>, !fir.ref<!fir.array<1xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>) -> tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<1xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>, !fir.ref<none>>
  ! CHECK:     %[[V_23:[0-9]+]] = fir.address_of(@default.nonTbpDefinedIoTable) : !fir.ref<tuple<i64, !fir.ref<!fir.array<0xtuple<!fir.ref<none>, !fir.ref<none>, i32, i1>>>, i1>>
  ! CHECK:     %[[V_24:[0-9]+]] = fir.convert %[[V_23]] : (!fir.ref<tuple<i64, !fir.ref<!fir.array<0xtuple<!fir.ref<none>, !fir.ref<none>, i32, i1>>>, i1>>) -> !fir.ref<none>
  ! CHECK:     %[[V_25:[0-9]+]] = fir.insert_value %[[V_22]], %[[V_24]], [3 : index] : (tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<1xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>, !fir.ref<none>>, !fir.ref<none>) -> tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<1xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>, !fir.ref<none>>
  ! CHECK:     fir.store %[[V_25]] to %[[V_16]] : !fir.ref<tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<1xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>, !fir.ref<none>>>
  ! CHECK:     %[[V_26:[0-9]+]] = fir.convert %[[V_16]] : (!fir.ref<tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<1xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>, !fir.ref<none>>>) -> !fir.ref<tuple<>>
  ! CHECK:     %[[V_27:[0-9]+]] = fir.call @_FortranAioInputNamelist(%[[V_7]], %[[V_26]]) fastmath<contract> : (!fir.ref<i8>, !fir.ref<tuple<>>) -> i1
  ! CHECK:     %[[V_28:[0-9]+]] = fir.call @_FortranAioEndIoStatement(%[[V_7]]) fastmath<contract> : (!fir.ref<i8>) -> i32
  namelist /rrr/ xxx
  read(*, rrr)
end

! CHECK-LABEL: c.func @_QPglobal_pointer
subroutine global_pointer
  real,pointer,save::ptrarray(:)
  ! CHECK:           %[[VAL_0:.*]] = arith.constant 1 : i64
  ! CHECK:           %[[VAL_2:.*]] = arith.constant 10 : i32
  ! CHECK:           %[[VAL_3:.*]] = fir.address_of(@_QFglobal_pointerEptrarray) : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  ! CHECK:           %[[VAL_4:.*]] = fir.declare %[[VAL_3]] {fortran_attrs = {{.*}}<pointer>, uniq_name = "_QFglobal_pointerEptrarray"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  ! CHECK:           %[[VAL_5:.*]] = fir.address_of(@_QQclX51b4dc7bdbe935748434745744d2a699) : !fir.ref<!fir.char<1,58>>
  ! CHECK:           %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (!fir.ref<!fir.char<1,58>>) -> !fir.ref<i8>
  ! CHECK:           %[[VAL_7:.*]] = fir.call @_FortranAioBeginExternalListOutput(%[[VAL_2]], %[[VAL_6]], %{{.*}}) fastmath<contract> : (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
  ! CHECK:           %[[VAL_8:.*]] = fir.alloca !fir.array<1xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>
  ! CHECK:           %[[VAL_9:.*]] = fir.undefined !fir.array<1xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>
  ! CHECK:           %[[VAL_10:.*]] = fir.address_of(@_QQclX707472617272617900) : !fir.ref<!fir.char<1,9>>
  ! CHECK:           %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (!fir.ref<!fir.char<1,9>>) -> !fir.ref<i8>
  ! CHECK:           %[[VAL_12:.*]] = fir.insert_value %[[VAL_9]], %[[VAL_11]], [0 : index, 0 : index] : (!fir.array<1xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>, !fir.ref<i8>) -> !fir.array<1xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>
  ! CHECK:           %[[VAL_13:.*]] = fir.convert %[[VAL_3]] : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK:           %[[VAL_14:.*]] = fir.insert_value %[[VAL_12]], %[[VAL_13]], [0 : index, 1 : index] : (!fir.array<1xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>, !fir.ref<!fir.box<none>>) -> !fir.array<1xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>
  ! CHECK:           fir.store %[[VAL_14]] to %[[VAL_8]] : !fir.ref<!fir.array<1xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>
  ! CHECK:           %[[VAL_15:.*]] = fir.alloca tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<1xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>, !fir.ref<none>>
  ! CHECK:           %[[VAL_16:.*]] = fir.undefined tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<1xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>, !fir.ref<none>>
  ! CHECK:           %[[VAL_17:.*]] = fir.address_of(@_QQclX6D7967726F757000) : !fir.ref<!fir.char<1,8>>
  ! CHECK:           %[[VAL_18:.*]] = fir.convert %[[VAL_17]] : (!fir.ref<!fir.char<1,8>>) -> !fir.ref<i8>
  ! CHECK:           %[[VAL_19:.*]] = fir.insert_value %[[VAL_16]], %[[VAL_18]], [0 : index] : (tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<1xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>, !fir.ref<none>>, !fir.ref<i8>) -> tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<1xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>, !fir.ref<none>>
  ! CHECK:           %[[VAL_20:.*]] = fir.insert_value %[[VAL_19]], %[[VAL_0]], [1 : index] : (tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<1xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>, !fir.ref<none>>, i64) -> tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<1xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>, !fir.ref<none>>
  ! CHECK:           %[[VAL_21:.*]] = fir.insert_value %[[VAL_20]], %[[VAL_8]], [2 : index] : (tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<1xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>, !fir.ref<none>>, !fir.ref<!fir.array<1xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>) -> tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<1xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>, !fir.ref<none>>
  ! CHECK:           %[[VAL_22:.*]] = fir.address_of(@default.nonTbpDefinedIoTable) : !fir.ref<tuple<i64, !fir.ref<!fir.array<0xtuple<!fir.ref<none>, !fir.ref<none>, i32, i1>>>, i1>>
  ! CHECK:           %[[VAL_23:.*]] = fir.convert %[[VAL_22]] : (!fir.ref<tuple<i64, !fir.ref<!fir.array<0xtuple<!fir.ref<none>, !fir.ref<none>, i32, i1>>>, i1>>) -> !fir.ref<none>
  ! CHECK:           %[[VAL_24:.*]] = fir.insert_value %[[VAL_21]], %[[VAL_23]], [3 : index] : (tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<1xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>, !fir.ref<none>>, !fir.ref<none>) -> tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<1xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>, !fir.ref<none>>
  ! CHECK:           fir.store %[[VAL_24]] to %[[VAL_15]] : !fir.ref<tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<1xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>, !fir.ref<none>>>
  ! CHECK:           %[[VAL_25:.*]] = fir.convert %[[VAL_15]] : (!fir.ref<tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<1xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>, !fir.ref<none>>>) -> !fir.ref<tuple<>>
  ! CHECK:           %[[VAL_26:.*]] = fir.call @_FortranAioOutputNamelist(%[[VAL_7]], %[[VAL_25]]) fastmath<contract> : (!fir.ref<i8>, !fir.ref<tuple<>>) -> i1
  ! CHECK:           %[[VAL_27:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_7]]) fastmath<contract> : (!fir.ref<i8>) -> i32
  namelist/mygroup/ptrarray
  write(10, nml=mygroup)
end

module mmm
  real rrr
  namelist /aaa/ rrr
end

! CHECK-LABEL: c.func @_QPrename_sub
subroutine rename_sub
  use mmm, bbb => aaa
  rrr = 3.
  ! CHECK:           %[[VAL_0:.*]] = arith.constant 1 : i64
  ! CHECK:           %[[VAL_2:.*]] = arith.constant 6 : i32
  ! CHECK:           %[[VAL_3:.*]] = arith.constant 3.000000e+00 : f32
  ! CHECK:           %[[VAL_4:.*]] = fir.alloca !fir.box<!fir.ptr<f32>>
  ! CHECK:           %[[VAL_5:.*]] = fir.address_of(@_QMmmmErrr) : !fir.ref<f32>
  ! CHECK:           %[[VAL_6:.*]] = fir.declare %[[VAL_5]] {uniq_name = "_QMmmmErrr"} : (!fir.ref<f32>) -> !fir.ref<f32>
  ! CHECK:           fir.store %[[VAL_3]] to %[[VAL_6]] : !fir.ref<f32>
  ! CHECK:           %[[VAL_7:.*]] = fir.address_of(@_QQclX51b4dc7bdbe935748434745744d2a699) : !fir.ref<!fir.char<1,58>>
  ! CHECK:           %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (!fir.ref<!fir.char<1,58>>) -> !fir.ref<i8>
  ! CHECK:           %[[VAL_9:.*]] = fir.call @_FortranAioBeginExternalListOutput(%[[VAL_2]], %[[VAL_8]], %{{.*}}) fastmath<contract> : (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
  ! CHECK:           %[[VAL_10:.*]] = fir.alloca !fir.array<1xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>
  ! CHECK:           %[[VAL_11:.*]] = fir.undefined !fir.array<1xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>
  ! CHECK:           %[[VAL_12:.*]] = fir.address_of(@_QQclX72727200) : !fir.ref<!fir.char<1,4>>
  ! CHECK:           %[[VAL_13:.*]] = fir.convert %[[VAL_12]] : (!fir.ref<!fir.char<1,4>>) -> !fir.ref<i8>
  ! CHECK:           %[[VAL_14:.*]] = fir.insert_value %[[VAL_11]], %[[VAL_13]], [0 : index, 0 : index] : (!fir.array<1xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>, !fir.ref<i8>) -> !fir.array<1xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>
  ! CHECK:           %[[VAL_15:.*]] = fir.embox %[[VAL_6]] : (!fir.ref<f32>) -> !fir.box<!fir.ptr<f32>>
  ! CHECK:           fir.store %[[VAL_15]] to %[[VAL_4]] : !fir.ref<!fir.box<!fir.ptr<f32>>>
  ! CHECK:           %[[VAL_16:.*]] = fir.convert %[[VAL_4]] : (!fir.ref<!fir.box<!fir.ptr<f32>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK:           %[[VAL_17:.*]] = fir.insert_value %[[VAL_14]], %[[VAL_16]], [0 : index, 1 : index] : (!fir.array<1xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>, !fir.ref<!fir.box<none>>) -> !fir.array<1xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>
  ! CHECK:           fir.store %[[VAL_17]] to %[[VAL_10]] : !fir.ref<!fir.array<1xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>
  ! CHECK:           %[[VAL_18:.*]] = fir.alloca tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<1xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>, !fir.ref<none>>
  ! CHECK:           %[[VAL_19:.*]] = fir.undefined tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<1xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>, !fir.ref<none>>
  ! CHECK:           %[[VAL_20:.*]] = fir.address_of(@_QQclX61616100) : !fir.ref<!fir.char<1,4>>
  ! CHECK:           %[[VAL_21:.*]] = fir.convert %[[VAL_20]] : (!fir.ref<!fir.char<1,4>>) -> !fir.ref<i8>
  ! CHECK:           %[[VAL_22:.*]] = fir.insert_value %[[VAL_19]], %[[VAL_21]], [0 : index] : (tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<1xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>, !fir.ref<none>>, !fir.ref<i8>) -> tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<1xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>, !fir.ref<none>>
  ! CHECK:           %[[VAL_23:.*]] = fir.insert_value %[[VAL_22]], %[[VAL_0]], [1 : index] : (tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<1xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>, !fir.ref<none>>, i64) -> tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<1xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>, !fir.ref<none>>
  ! CHECK:           %[[VAL_24:.*]] = fir.insert_value %[[VAL_23]], %[[VAL_10]], [2 : index] : (tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<1xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>, !fir.ref<none>>, !fir.ref<!fir.array<1xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>) -> tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<1xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>, !fir.ref<none>>
  ! CHECK:           %[[VAL_25:.*]] = fir.address_of(@default.nonTbpDefinedIoTable) : !fir.ref<tuple<i64, !fir.ref<!fir.array<0xtuple<!fir.ref<none>, !fir.ref<none>, i32, i1>>>, i1>>
  ! CHECK:           %[[VAL_26:.*]] = fir.convert %[[VAL_25]] : (!fir.ref<tuple<i64, !fir.ref<!fir.array<0xtuple<!fir.ref<none>, !fir.ref<none>, i32, i1>>>, i1>>) -> !fir.ref<none>
  ! CHECK:           %[[VAL_27:.*]] = fir.insert_value %[[VAL_24]], %[[VAL_26]], [3 : index] : (tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<1xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>, !fir.ref<none>>, !fir.ref<none>) -> tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<1xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>, !fir.ref<none>>
  ! CHECK:           fir.store %[[VAL_27]] to %[[VAL_18]] : !fir.ref<tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<1xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>, !fir.ref<none>>>
  ! CHECK:           %[[VAL_28:.*]] = fir.convert %[[VAL_18]] : (!fir.ref<tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<1xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>, !fir.ref<none>>>) -> !fir.ref<tuple<>>
  ! CHECK:           %[[VAL_29:.*]] = fir.call @_FortranAioOutputNamelist(%[[VAL_9]], %[[VAL_28]]) fastmath<contract> : (!fir.ref<i8>, !fir.ref<tuple<>>) -> i1
  ! CHECK:           %[[VAL_30:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_9]]) fastmath<contract> : (!fir.ref<i8>) -> i32
  write(*,bbb)
end

! CHECK-NOT:   bbb
! CHECK:       fir.string_lit "aaa\00"(4) : !fir.char<1,4>

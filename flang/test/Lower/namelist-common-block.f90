! RUN: bbc -emit-hlfir -o - %s | FileCheck %s

! Test that allocatable or pointer item from namelist are retrieved correctly
! if they are part of a common block as well.

program nml_common
  integer :: i
  real, pointer :: p(:)
  namelist /t/i,p
  common /c/i,p
  
  allocate(p(2))
  call print_t()
contains
  subroutine print_t()
    write(*,t)
  end subroutine
end

! CHECK-LABEL:   func.func private @_QFPprint_t() attributes {fir.host_symbol = @_QQmain, llvm.linkage = #{{.*}}<internal>} {
! CHECK:           %[[VAL_0:.*]] = fir.alloca !fir.box<!fir.ptr<i32>>
! CHECK:           %[[VAL_1:.*]] = fir.address_of(@c_) : !fir.ref<!fir.array<56xi8>>
! CHECK:           %[[VAL_2:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.array<56xi8>>) -> !fir.ref<!fir.array<?xi8>>
! CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_4:.*]] = fir.coordinate_of %[[VAL_2]], %[[VAL_3]] : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
! CHECK:           %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (!fir.ref<i8>) -> !fir.ref<i32>
! CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_5]] {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_7:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.array<56xi8>>) -> !fir.ref<!fir.array<?xi8>>
! CHECK:           %[[VAL_8:.*]] = arith.constant 8 : index
! CHECK:           %[[VAL_9:.*]] = fir.coordinate_of %[[VAL_7]], %[[VAL_8]] : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
! CHECK:           %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (!fir.ref<i8>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! CHECK:           %[[VAL_11:.*]]:2 = hlfir.declare %[[VAL_10]] {fortran_attrs = #{{.*}}<pointer>, uniq_name = "_QFEp"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>)
! CHECK:           %[[VAL_12:.*]] = arith.constant 6 : i32
! CHECK:           %[[VAL_13:.*]] = fir.address_of(@_QQclXcb818d0ca98c80cc21b98226ec6d98a0) : !fir.ref<!fir.char<1,71>>
! CHECK:           %[[VAL_14:.*]] = fir.convert %[[VAL_13]] : (!fir.ref<!fir.char<1,71>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_15:.*]] = arith.constant 16 : i32
! CHECK:           %[[VAL_16:.*]] = fir.call @_FortranAioBeginExternalListOutput(%[[VAL_12]], %[[VAL_14]], %[[VAL_15]]) fastmath<contract> : (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:           %[[VAL_17:.*]] = fir.alloca !fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>
! CHECK:           %[[VAL_18:.*]] = fir.undefined !fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>
! CHECK:           %[[VAL_19:.*]] = fir.address_of(@_QQclX6900) : !fir.ref<!fir.char<1,2>>
! CHECK:           %[[VAL_20:.*]] = fir.convert %[[VAL_19]] : (!fir.ref<!fir.char<1,2>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_21:.*]] = fir.insert_value %[[VAL_18]], %[[VAL_20]], [0 : index, 0 : index] : (!fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>, !fir.ref<i8>) -> !fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>
! CHECK:           %[[VAL_22:.*]] = fir.embox %[[VAL_6]]#1 : (!fir.ref<i32>) -> !fir.box<!fir.ptr<i32>>
! CHECK:           fir.store %[[VAL_22]] to %[[VAL_0]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:           %[[VAL_23:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> !fir.ref<!fir.box<none>>
! CHECK:           %[[VAL_24:.*]] = fir.insert_value %[[VAL_21]], %[[VAL_23]], [0 : index, 1 : index] : (!fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>, !fir.ref<!fir.box<none>>) -> !fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>
! CHECK:           %[[VAL_25:.*]] = fir.address_of(@_QQclX7000) : !fir.ref<!fir.char<1,2>>
! CHECK:           %[[VAL_26:.*]] = fir.convert %[[VAL_25]] : (!fir.ref<!fir.char<1,2>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_27:.*]] = fir.insert_value %[[VAL_24]], %[[VAL_26]], [1 : index, 0 : index] : (!fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>, !fir.ref<i8>) -> !fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>
! CHECK:           %[[VAL_28:.*]] = fir.address_of(@c_) : !fir.ref<!fir.array<56xi8>>
! CHECK:           %[[VAL_29:.*]] = fir.convert %[[VAL_28]] : (!fir.ref<!fir.array<56xi8>>) -> !fir.ref<!fir.array<?xi8>>
! CHECK:           %[[VAL_30:.*]] = arith.constant 8 : index
! CHECK:           %[[VAL_31:.*]] = fir.coordinate_of %[[VAL_29]], %[[VAL_30]] : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
! CHECK:           %[[VAL_32:.*]] = fir.convert %[[VAL_31]] : (!fir.ref<i8>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! CHECK:           %[[VAL_33:.*]] = fir.convert %[[VAL_32]] : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:           %[[VAL_34:.*]] = fir.insert_value %[[VAL_27]], %[[VAL_33]], [1 : index, 1 : index] : (!fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>, !fir.ref<!fir.box<none>>) -> !fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>
! CHECK:           fir.store %[[VAL_34]] to %[[VAL_17]] : !fir.ref<!fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>
! CHECK:           %[[VAL_35:.*]] = fir.alloca tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>, !fir.ref<none>>
! CHECK:           %[[VAL_36:.*]] = fir.undefined tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>, !fir.ref<none>>
! CHECK:           %[[VAL_37:.*]] = fir.address_of(@_QQclX7400) : !fir.ref<!fir.char<1,2>>
! CHECK:           %[[VAL_38:.*]] = fir.convert %[[VAL_37]] : (!fir.ref<!fir.char<1,2>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_39:.*]] = fir.insert_value %[[VAL_36]], %[[VAL_38]], [0 : index] : (tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>, !fir.ref<none>>, !fir.ref<i8>) -> tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>, !fir.ref<none>>
! CHECK:           %[[VAL_40:.*]] = arith.constant 2 : i64
! CHECK:           %[[VAL_41:.*]] = fir.insert_value %[[VAL_39]], %[[VAL_40]], [1 : index] : (tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>, !fir.ref<none>>, i64) -> tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>, !fir.ref<none>>
! CHECK:           %[[VAL_42:.*]] = fir.insert_value %[[VAL_41]], %[[VAL_17]], [2 : index] : (tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>, !fir.ref<none>>, !fir.ref<!fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>) -> tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>, !fir.ref<none>>
! CHECK:           %[[VAL_43:.*]] = fir.address_of(@default.nonTbpDefinedIoTable) : !fir.ref<tuple<i64, !fir.ref<!fir.array<0xtuple<!fir.ref<none>, !fir.ref<none>, i32, i1>>>, i1>>
! CHECK:           %[[VAL_44:.*]] = fir.convert %[[VAL_43]] : (!fir.ref<tuple<i64, !fir.ref<!fir.array<0xtuple<!fir.ref<none>, !fir.ref<none>, i32, i1>>>, i1>>) -> !fir.ref<none>
! CHECK:           %[[VAL_45:.*]] = fir.insert_value %[[VAL_42]], %[[VAL_44]], [3 : index] : (tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>, !fir.ref<none>>, !fir.ref<none>) -> tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>, !fir.ref<none>>
! CHECK:           fir.store %[[VAL_45]] to %[[VAL_35]] : !fir.ref<tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>, !fir.ref<none>>>
! CHECK:           %[[VAL_46:.*]] = fir.convert %[[VAL_35]] : (!fir.ref<tuple<!fir.ref<i8>, i64, !fir.ref<!fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>>, !fir.ref<none>>>) -> !fir.ref<tuple<>>
! CHECK:           %[[VAL_47:.*]] = fir.call @_FortranAioOutputNamelist(%[[VAL_16]], %[[VAL_46]]) fastmath<contract> : (!fir.ref<i8>, !fir.ref<tuple<>>) -> i1
! CHECK:           %[[VAL_48:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_16]]) fastmath<contract> : (!fir.ref<i8>) -> i32

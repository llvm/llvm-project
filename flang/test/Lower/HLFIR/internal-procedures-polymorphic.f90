! Test lowering of internal procedure capturing OPTIONAL polymorphic
! objects.
! RUN: bbc -emit-hlfir --polymorphic-type -o - %s -I nw | FileCheck %s


module captured_optional_polymorphic
  type sometype
  end type
contains
subroutine test(x, y)
  class(sometype), optional :: x
  class(sometype), optional :: y(2:)
  call internal()
contains
  subroutine internal()
    if (present(x).and.present(y)) then
      print *, same_type_as(x, y)
    end if
  end subroutine
end
end module

! CHECK-LABEL:   func.func @_QMcaptured_optional_polymorphicPtest(
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare{{.*}}Ex
! CHECK:           %[[VAL_3:.*]] = arith.constant 2 : i64
! CHECK:           %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (i64) -> index
! CHECK:           %[[VAL_5:.*]] = fir.shift %[[VAL_4]] : (index) -> !fir.shift<1>
! CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare{{.*}}Ey
! CHECK:           %[[VAL_7:.*]] = fir.alloca tuple<!fir.class<!fir.type<_QMcaptured_optional_polymorphicTsometype>>, !fir.class<!fir.array<?x!fir.type<_QMcaptured_optional_polymorphicTsometype>>>>
! CHECK:           %[[VAL_8:.*]] = arith.constant 0 : i32
! CHECK:           %[[VAL_9:.*]] = fir.coordinate_of %[[VAL_7]], %[[VAL_8]]
! CHECK:           %[[VAL_10:.*]] = fir.is_present %[[VAL_2]]#1 : (!fir.class<!fir.type<_QMcaptured_optional_polymorphicTsometype>>) -> i1
! CHECK:           fir.if %[[VAL_10]] {
! CHECK:             fir.store %[[VAL_2]]#1 to %[[VAL_9]] : !fir.ref<!fir.class<!fir.type<_QMcaptured_optional_polymorphicTsometype>>>
! CHECK:           } else {
! CHECK:             %[[VAL_11:.*]] = fir.zero_bits !fir.ref<!fir.type<_QMcaptured_optional_polymorphicTsometype>>
! CHECK:             %[[VAL_12:.*]] = fir.embox %[[VAL_11]] : (!fir.ref<!fir.type<_QMcaptured_optional_polymorphicTsometype>>) -> !fir.class<!fir.type<_QMcaptured_optional_polymorphicTsometype>>
! CHECK:             fir.store %[[VAL_12]] to %[[VAL_9]] : !fir.ref<!fir.class<!fir.type<_QMcaptured_optional_polymorphicTsometype>>>
! CHECK:           }
! CHECK:           %[[VAL_13:.*]] = arith.constant 1 : i32
! CHECK:           %[[VAL_14:.*]] = fir.coordinate_of %[[VAL_7]], %[[VAL_13]]
! CHECK:           %[[VAL_15:.*]] = fir.is_present %[[VAL_6]]#1 : (!fir.class<!fir.array<?x!fir.type<_QMcaptured_optional_polymorphicTsometype>>>) -> i1
! CHECK:           fir.if %[[VAL_15]] {
! CHECK:             %[[VAL_16:.*]] = fir.shift %[[VAL_4]] : (index) -> !fir.shift<1>
! CHECK:             %[[VAL_17:.*]] = fir.rebox %[[VAL_6]]#1(%[[VAL_16]]) : (!fir.class<!fir.array<?x!fir.type<_QMcaptured_optional_polymorphicTsometype>>>, !fir.shift<1>) -> !fir.class<!fir.array<?x!fir.type<_QMcaptured_optional_polymorphicTsometype>>>
! CHECK:             fir.store %[[VAL_17]] to %[[VAL_14]] : !fir.ref<!fir.class<!fir.array<?x!fir.type<_QMcaptured_optional_polymorphicTsometype>>>>
! CHECK:           } else {
! CHECK:             %[[VAL_18:.*]] = fir.type_desc !fir.type<_QMcaptured_optional_polymorphicTsometype>
! CHECK:             %[[VAL_19:.*]] = fir.convert %[[VAL_14]] : (!fir.ref<!fir.class<!fir.array<?x!fir.type<_QMcaptured_optional_polymorphicTsometype>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:             %[[VAL_20:.*]] = fir.convert %[[VAL_18]] : (!fir.tdesc<!fir.type<_QMcaptured_optional_polymorphicTsometype>>) -> !fir.ref<none>
! CHECK:             %[[VAL_21:.*]] = arith.constant 1 : i32
! CHECK:             %[[VAL_22:.*]] = arith.constant 0 : i32
! CHECK:             %[[VAL_23:.*]] = fir.call @_FortranAPointerNullifyDerived(%[[VAL_19]], %[[VAL_20]], %[[VAL_21]], %[[VAL_22]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.ref<none>, i32, i32) -> none
! CHECK:           }
! CHECK:           fir.call @_QMcaptured_optional_polymorphicFtestPinternal(%[[VAL_7]])

! CHECK-LABEL: func.func{{.*}} @_QMcaptured_optional_polymorphicFtestPinternal(
! CHECK-SAME:      %[[VAL_0:.*]]: !fir.ref<tuple<{{.*}}>>
! CHECK:           %[[VAL_1:.*]] = arith.constant 0 : i32
! CHECK:           %[[VAL_2:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_1]]
! CHECK:           %[[VAL_3:.*]] = fir.load %[[VAL_2]] : !fir.ref<!fir.class<!fir.type<_QMcaptured_optional_polymorphicTsometype>>>
! CHECK:           %[[VAL_4:.*]] = fir.box_addr %[[VAL_3]] : (!fir.class<!fir.type<_QMcaptured_optional_polymorphicTsometype>>) -> !fir.ref<!fir.type<_QMcaptured_optional_polymorphicTsometype>>
! CHECK:           %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (!fir.ref<!fir.type<_QMcaptured_optional_polymorphicTsometype>>) -> i64
! CHECK:           %[[VAL_6:.*]] = arith.constant 0 : i64
! CHECK:           %[[VAL_7:.*]] = arith.cmpi ne, %[[VAL_5]], %[[VAL_6]] : i64
! CHECK:           %[[VAL_8:.*]] = fir.absent !fir.class<!fir.type<_QMcaptured_optional_polymorphicTsometype>>
! CHECK:           %[[VAL_9:.*]] = arith.select %[[VAL_7]], %[[VAL_3]], %[[VAL_8]] : !fir.class<!fir.type<_QMcaptured_optional_polymorphicTsometype>>
! CHECK:           %[[VAL_10:.*]]:2 = hlfir.declare %[[VAL_9]] {fortran_attrs = #fir.var_attrs<optional, host_assoc>, {{.*}}Ex
! CHECK:           %[[VAL_11:.*]] = arith.constant 1 : i32
! CHECK:           %[[VAL_12:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_11]]
! CHECK:           %[[VAL_13:.*]] = fir.load %[[VAL_12]] : !fir.ref<!fir.class<!fir.array<?x!fir.type<_QMcaptured_optional_polymorphicTsometype>>>>
! CHECK:           %[[VAL_14:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_15:.*]]:3 = fir.box_dims %[[VAL_13]], %[[VAL_14]]
! CHECK:           %[[VAL_16:.*]] = fir.box_addr %[[VAL_13]]
! CHECK:           %[[VAL_17:.*]] = fir.convert %[[VAL_16]] : (!fir.ref<!fir.array<?x!fir.type<_QMcaptured_optional_polymorphicTsometype>>>) -> i64
! CHECK:           %[[VAL_18:.*]] = arith.constant 0 : i64
! CHECK:           %[[VAL_19:.*]] = arith.cmpi ne, %[[VAL_17]], %[[VAL_18]] : i64
! CHECK:           %[[VAL_20:.*]] = fir.absent !fir.class<!fir.array<?x!fir.type<_QMcaptured_optional_polymorphicTsometype>>>
! CHECK:           %[[VAL_21:.*]] = arith.select %[[VAL_19]], %[[VAL_13]], %[[VAL_20]] : !fir.class<!fir.array<?x!fir.type<_QMcaptured_optional_polymorphicTsometype>>>
! CHECK:           %[[VAL_22:.*]] = fir.shift %[[VAL_15]]#0 : (index) -> !fir.shift<1>
! CHECK:           %[[VAL_23:.*]]:2 = hlfir.declare %[[VAL_21]](%[[VAL_22]]) {fortran_attrs = #fir.var_attrs<optional, host_assoc>, {{.*}}Ey

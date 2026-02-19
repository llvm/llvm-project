! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: fir.global @_QMc_interoperability_testEthis_thing : !fir.type<_QMc_interoperability_testTthing_with_pointer{cptr:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}> {
! CHECK:         %[[VAL_0:.*]] = fir.undefined !fir.type<_QMc_interoperability_testTthing_with_pointer{cptr:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>
! CHECK:         %[[VAL_1:.*]] = fir.undefined !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK:         %[[VAL_2:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK:         %[[VAL_3:.*]] = arith.constant 0 : i64
! CHECK:         %[[VAL_4:.*]] = fir.insert_value %[[VAL_1]], %[[VAL_3]], ["__address", !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>] : (!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>, i64) -> !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK:         %[[VAL_5:.*]] = fir.field_index cptr, !fir.type<_QMc_interoperability_testTthing_with_pointer{cptr:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>
! CHECK:         %[[VAL_6:.*]] = fir.insert_value %[[VAL_0]], %[[VAL_4]], ["cptr", !fir.type<_QMc_interoperability_testTthing_with_pointer{cptr:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>] : (!fir.type<_QMc_interoperability_testTthing_with_pointer{cptr:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>) -> !fir.type<_QMc_interoperability_testTthing_with_pointer{cptr:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>
! CHECK:         fir.has_value %[[VAL_6]] : !fir.type<_QMc_interoperability_testTthing_with_pointer{cptr:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>
! CHECK:       }

! CHECK-LABEL: func @_QMc_interoperability_testPget_a_thing() -> !fir.type<_QMc_interoperability_testTthing_with_pointer{cptr:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}> {
! CHECK:         %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:         %[[VAL_1:.*]] = fir.address_of(@_QM__fortran_builtinsEC__builtin_c_null_ptr) : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>
! CHECK:         %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QM__fortran_builtinsEC__builtin_c_null_ptr"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>) -> (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>)
! CHECK:         %[[VAL_3:.*]] = fir.address_of(@_QMc_interoperability_testEthis_thing) : !fir.ref<!fir.type<_QMc_interoperability_testTthing_with_pointer{cptr:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>
! CHECK:         %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_3]] {uniq_name = "_QMc_interoperability_testEthis_thing"} : (!fir.ref<!fir.type<_QMc_interoperability_testTthing_with_pointer{cptr:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>) -> (!fir.ref<!fir.type<_QMc_interoperability_testTthing_with_pointer{cptr:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>, !fir.ref<!fir.type<_QMc_interoperability_testTthing_with_pointer{cptr:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>)
! CHECK:         %[[VAL_5:.*]] = fir.alloca !fir.type<_QMc_interoperability_testTthing_with_pointer{cptr:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}> {bindc_name = "get_a_thing", uniq_name = "_QMc_interoperability_testFget_a_thingEget_a_thing"}
! CHECK:         %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_5]] {uniq_name = "_QMc_interoperability_testFget_a_thingEget_a_thing"} : (!fir.ref<!fir.type<_QMc_interoperability_testTthing_with_pointer{cptr:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>) -> (!fir.ref<!fir.type<_QMc_interoperability_testTthing_with_pointer{cptr:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>, !fir.ref<!fir.type<_QMc_interoperability_testTthing_with_pointer{cptr:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>)
! CHECK:         %[[VAL_7:.*]] = fir.address_of(@_QQ_QMc_interoperability_testTthing_with_pointer.DerivedInit) : !fir.ref<!fir.type<_QMc_interoperability_testTthing_with_pointer{cptr:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>
! CHECK:         fir.copy %[[VAL_7]] to %[[VAL_6]]#0 no_overlap : !fir.ref<!fir.type<_QMc_interoperability_testTthing_with_pointer{cptr:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>, !fir.ref<!fir.type<_QMc_interoperability_testTthing_with_pointer{cptr:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>
! CHECK:         hlfir.assign %[[VAL_4]]#0 to %[[VAL_6]]#0 : !fir.ref<!fir.type<_QMc_interoperability_testTthing_with_pointer{cptr:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>, !fir.ref<!fir.type<_QMc_interoperability_testTthing_with_pointer{cptr:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>
! CHECK:         %[[VAL_8:.*]] = fir.load %[[VAL_6]]#0 : !fir.ref<!fir.type<_QMc_interoperability_testTthing_with_pointer{cptr:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>
! CHECK:         return %[[VAL_8]] : !fir.type<_QMc_interoperability_testTthing_with_pointer{cptr:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>
! CHECK:       }

module c_interoperability_test
  use iso_c_binding, only: c_ptr, c_null_ptr

  type thing_with_pointer
     type(c_ptr) :: cptr = c_null_ptr
  end type thing_with_pointer

  type(thing_with_pointer) :: this_thing

contains
  function get_a_thing()
    type(thing_with_pointer) :: get_a_thing
    get_a_thing = this_thing
  end function get_a_thing
end module c_interoperability_test

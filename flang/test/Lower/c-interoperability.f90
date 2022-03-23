! RUN: bbc %s -o - | FileCheck %s

! CHECK-LABEL: fir.global @_QMc_interoperability_testEthis_thing : !fir.type<_QMc_interoperability_testTthing_with_pointer{cptr:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}> {
! CHECK:         %[[VAL_0:.*]] = arith.constant 0 : i64
! CHECK:         %[[VAL_1:.*]] = fir.undefined !fir.type<_QMc_interoperability_testTthing_with_pointer{cptr:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>
! CHECK:         %[[VAL_2:.*]] = fir.undefined !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK:         %[[VAL_3:.*]] = fir.insert_value %[[VAL_2]], %[[VAL_0]], ["__address", !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>] : (!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>, i64) -> !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK:         %[[VAL_4:.*]] = fir.insert_value %[[VAL_1]], %[[VAL_3]], ["cptr", !fir.type<_QMc_interoperability_testTthing_with_pointer{cptr:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>] : (!fir.type<_QMc_interoperability_testTthing_with_pointer{cptr:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>) -> !fir.type<_QMc_interoperability_testTthing_with_pointer{cptr:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>
! CHECK:         fir.has_value %[[VAL_4]] : !fir.type<_QMc_interoperability_testTthing_with_pointer{cptr:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>
! CHECK:       }

! CHECK-LABEL: func @_QMc_interoperability_testPget_a_thing() -> !fir.type<_QMc_interoperability_testTthing_with_pointer{cptr:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}> {
! CHECK:         %[[VAL_2:.*]] = fir.alloca !fir.box<!fir.type<_QMc_interoperability_testTthing_with_pointer{cptr:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>
! CHECK:         %[[VAL_3:.*]] = fir.address_of(@_QMc_interoperability_testEthis_thing) : !fir.ref<!fir.type<_QMc_interoperability_testTthing_with_pointer{cptr:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>
! CHECK:         %[[VAL_4:.*]] = fir.alloca !fir.type<_QMc_interoperability_testTthing_with_pointer{cptr:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}> {bindc_name = "get_a_thing", uniq_name = "_QMc_interoperability_testFget_a_thingEget_a_thing"}
! CHECK:         %[[VAL_5:.*]] = fir.embox %[[VAL_4]] : (!fir.ref<!fir.type<_QMc_interoperability_testTthing_with_pointer{cptr:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>) -> !fir.box<!fir.type<_QMc_interoperability_testTthing_with_pointer{cptr:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>
! CHECK:         %[[VAL_10:.*]] = fir.embox %[[VAL_3]] : (!fir.ref<!fir.type<_QMc_interoperability_testTthing_with_pointer{cptr:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>) -> !fir.box<!fir.type<_QMc_interoperability_testTthing_with_pointer{cptr:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>
! CHECK:         return %{{.*}} : !fir.type<_QMc_interoperability_testTthing_with_pointer{cptr:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>
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

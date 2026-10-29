! RUN: %flang_fc1 -emit-hlfir -fcoarray %s -o - | FileCheck %s --check-prefixes=COARRAY
! RUN: not %flang_fc1 -emit-hlfir %s 2>&1 | FileCheck %s --check-prefixes=NOCOARRAY

! NOCOARRAY: Not yet implemented: Multi-image features are experimental and are disabled by default, use '-fcoarray' to enable.

subroutine query_base()
  use iso_fortran_env, only: event_type
  type(event_type), save :: ev[*]
  integer :: count
  call event_query(ev, count)
end subroutine

! COARRAY-LABEL: func.func @_QPquery_base() {
! COARRAY: %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
! COARRAY: %[[VAL_1:.*]] = fir.alloca i32 <{bindc_name = "count", uniq_name = "_QFquery_baseEcount"}>
! COARRAY: %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1]] uniq_name("_QFquery_baseEcount") : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! COARRAY: %[[VAL_3:.*]] = fir.address_of(@_QFquery_baseEev) : !fir.ref<!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>, corank:1>>
! COARRAY: %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_3]] uniq_name("_QFquery_baseEev") : {{.*}}
! COARRAY: %[[VAL_5:.*]] = fir.load %[[VAL_4]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>, corank:1>>
! COARRAY: %[[VAL_6:.*]] = fir.box_addr %[[VAL_5]] : ({{.*}}) -> !fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>
! COARRAY: %[[VAL_7:.*]] = mif.event_query %[[VAL_6]] : (!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>) -> i32
! COARRAY: fir.store %[[VAL_7]] to %[[VAL_2]]#0 : !fir.ref<i32>
! COARRAY: return
! COARRAY: }


subroutine query_with_stat()
  use iso_fortran_env, only: event_type
  type(event_type), save :: ev[*]
  integer :: count, stat
  call event_query(ev, count, stat)
end subroutine

! COARRAY-LABEL: func.func @_QPquery_with_stat() {
! COARRAY: %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
! COARRAY: %[[VAL_1:.*]] = fir.alloca i32 <{bindc_name = "count", uniq_name = "_QFquery_with_statEcount"}>
! COARRAY: %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1]] uniq_name("_QFquery_with_statEcount") : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! COARRAY: %[[VAL_3:.*]] = fir.address_of(@_QFquery_with_statEev) : !fir.ref<!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>, corank:1>>
! COARRAY: %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_3]] uniq_name("_QFquery_with_statEev") : {{.*}}
! COARRAY: %[[VAL_5:.*]] = fir.alloca i32 <{bindc_name = "stat", uniq_name = "_QFquery_with_statEstat"}>
! COARRAY: %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_5]] uniq_name("_QFquery_with_statEstat") : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! COARRAY: %[[VAL_7:.*]] = fir.load %[[VAL_4]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>, corank:1>>
! COARRAY: %[[VAL_8:.*]] = fir.box_addr %[[VAL_7]] : ({{.*}}) -> !fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>
! COARRAY: %[[VAL_9:.*]] = mif.event_query %[[VAL_8]] stat %[[VAL_6]]#0 : (!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>, !fir.ref<i32>) -> i32
! COARRAY: fir.store %[[VAL_9]] to %[[VAL_2]]#0 : !fir.ref<i32>
! COARRAY: return
! COARRAY: }

subroutine query_with_kinds()
  use iso_fortran_env, only: event_type, int16, int32, int64
  type(event_type), save :: ev[*]
  integer(int32) :: c4
  integer(int64) :: c8
  integer(int16) :: s2
  integer(int32) :: s4
  integer(int64) :: s8
  call event_query(ev, c4)
  call event_query(ev, c8)
  call event_query(ev, c4, s2)
  call event_query(ev, c4, s4)
  call event_query(ev, c8, s8)
end subroutine

! COARRAY-LABEL: func.func @_QPquery_with_kinds() {
! COARRAY: %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
! COARRAY: %[[VAL_1:.*]] = fir.alloca i32 <{bindc_name = "c4", uniq_name = "_QFquery_with_kindsEc4"}>
! COARRAY: %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1]] uniq_name("_QFquery_with_kindsEc4") : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! COARRAY: %[[VAL_3:.*]] = fir.alloca i64 <{bindc_name = "c8", uniq_name = "_QFquery_with_kindsEc8"}>
! COARRAY: %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_3]] uniq_name("_QFquery_with_kindsEc8") : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
! COARRAY: %[[VAL_5:.*]] = fir.address_of(@_QFquery_with_kindsEev) : !fir.ref<!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>, corank:1>>
! COARRAY: %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_5]] uniq_name("_QFquery_with_kindsEev") : {{.*}}
! COARRAY: %[[VAL_7:.*]] = fir.address_of(@_QMiso_fortran_env_implECint16) : !fir.ref<i32>
! COARRAY: %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_7]] uniq_name("_QMiso_fortran_env_implECint16") fortran_attrs<parameter> : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! COARRAY: %[[VAL_9:.*]] = fir.address_of(@_QMiso_fortran_env_implECint32) : !fir.ref<i32>
! COARRAY: %[[VAL_10:.*]]:2 = hlfir.declare %[[VAL_9]] uniq_name("_QMiso_fortran_env_implECint32") fortran_attrs<parameter> : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! COARRAY: %[[VAL_11:.*]] = fir.address_of(@_QMiso_fortran_env_implECint64) : !fir.ref<i32>
! COARRAY: %[[VAL_12:.*]]:2 = hlfir.declare %[[VAL_11]] uniq_name("_QMiso_fortran_env_implECint64") fortran_attrs<parameter> : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! COARRAY: %[[VAL_13:.*]] = fir.alloca i16 <{bindc_name = "s2", uniq_name = "_QFquery_with_kindsEs2"}>
! COARRAY: %[[VAL_14:.*]]:2 = hlfir.declare %[[VAL_13]] uniq_name("_QFquery_with_kindsEs2") : (!fir.ref<i16>) -> (!fir.ref<i16>, !fir.ref<i16>)
! COARRAY: %[[VAL_15:.*]] = fir.alloca i32 <{bindc_name = "s4", uniq_name = "_QFquery_with_kindsEs4"}>
! COARRAY: %[[VAL_16:.*]]:2 = hlfir.declare %[[VAL_15]] uniq_name("_QFquery_with_kindsEs4") : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! COARRAY: %[[VAL_17:.*]] = fir.alloca i64 <{bindc_name = "s8", uniq_name = "_QFquery_with_kindsEs8"}>
! COARRAY: %[[VAL_18:.*]]:2 = hlfir.declare %[[VAL_17]] uniq_name("_QFquery_with_kindsEs8") : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
! COARRAY: %[[VAL_19:.*]] = fir.load %[[VAL_6]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>, corank:1>>
! COARRAY: %[[VAL_20:.*]] = fir.box_addr %[[VAL_19]] : ({{.*}}) -> !fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>
! COARRAY: %[[VAL_21:.*]] = mif.event_query %[[VAL_20]] : (!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>) -> i32
! COARRAY: fir.store %[[VAL_21]] to %[[VAL_2]]#0 : !fir.ref<i32>
! COARRAY: %[[VAL_22:.*]] = fir.load %[[VAL_6]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>, corank:1>>
! COARRAY: %[[VAL_23:.*]] = fir.box_addr %[[VAL_22]] : ({{.*}}) -> !fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>
! COARRAY: %[[VAL_24:.*]] = mif.event_query %[[VAL_23]] : (!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>) -> i64
! COARRAY: fir.store %[[VAL_24]] to %[[VAL_4]]#0 : !fir.ref<i64>
! COARRAY: %[[VAL_25:.*]] = fir.load %[[VAL_6]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>, corank:1>>
! COARRAY: %[[VAL_26:.*]] = fir.box_addr %[[VAL_25]] : ({{.*}}) -> !fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>
! COARRAY: %[[VAL_27:.*]] = mif.event_query %[[VAL_26]] stat %[[VAL_14]]#0 : (!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>, !fir.ref<i16>) -> i32
! COARRAY: fir.store %[[VAL_27]] to %[[VAL_2]]#0 : !fir.ref<i32>
! COARRAY: %[[VAL_28:.*]] = fir.load %[[VAL_6]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>, corank:1>>
! COARRAY: %[[VAL_29:.*]] = fir.box_addr %[[VAL_28]] : ({{.*}}) -> !fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>
! COARRAY: %[[VAL_30:.*]] = mif.event_query %[[VAL_29]] stat %[[VAL_16]]#0 : (!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>, !fir.ref<i32>) -> i32
! COARRAY: fir.store %[[VAL_30]] to %[[VAL_2]]#0 : !fir.ref<i32>
! COARRAY: %[[VAL_31:.*]] = fir.load %[[VAL_6]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>, corank:1>>
! COARRAY: %[[VAL_32:.*]] = fir.box_addr %[[VAL_31]] : ({{.*}}) -> !fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>
! COARRAY: %[[VAL_33:.*]] = mif.event_query %[[VAL_32]] stat %[[VAL_18]]#0 : (!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>, !fir.ref<i64>) -> i64
! COARRAY: fir.store %[[VAL_33]] to %[[VAL_4]]#0 : !fir.ref<i64>
! COARRAY: return
! COARRAY: }

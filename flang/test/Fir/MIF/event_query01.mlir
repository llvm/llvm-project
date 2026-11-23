// RUN: fir-opt --mif-convert %s | FileCheck %s


func.func @_QPquery_base() {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.alloca i32 <{bindc_name = "count", uniq_name = "_QFquery_baseEcount"}>
  %2:2 = hlfir.declare %1 uniq_name("_QFquery_baseEcount") : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %3 = fir.address_of(@_QFquery_baseEev) : !fir.ref<!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>>
  %4:2 = hlfir.declare %3 uniq_name("_QFquery_baseEev") : (!fir.ref<!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>>, !fir.ref<!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>>)
  %5 = fir.load %4#0 : !fir.ref<!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>>
  %6 = fir.box_addr %5 : (!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>) -> !fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>
  %7 = mif.event_query %6 : (!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>) -> i32
  fir.store %7 to %2#0 : !fir.ref<i32>
  return
}
func.func @_QPquery_with_stat() {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.alloca i32 <{bindc_name = "count", uniq_name = "_QFquery_with_statEcount"}>
  %2:2 = hlfir.declare %1 uniq_name("_QFquery_with_statEcount") : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %3 = fir.address_of(@_QFquery_with_statEev) : !fir.ref<!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>>
  %4:2 = hlfir.declare %3 uniq_name("_QFquery_with_statEev") : (!fir.ref<!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>>, !fir.ref<!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>>)
  %5 = fir.alloca i32 <{bindc_name = "stat", uniq_name = "_QFquery_with_statEstat"}>
  %6:2 = hlfir.declare %5 uniq_name("_QFquery_with_statEstat") : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %7 = fir.load %4#0 : !fir.ref<!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>>
  %8 = fir.box_addr %7 : (!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>) -> !fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>
  %9 = mif.event_query %8 stat %6#0 : (!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, !fir.ref<i32>) -> i32
  fir.store %9 to %2#0 : !fir.ref<i32>
  return
}
func.func @_QPquery_with_kinds() {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.alloca i32 <{bindc_name = "c4", uniq_name = "_QFquery_with_kindsEc4"}>
  %2:2 = hlfir.declare %1 uniq_name("_QFquery_with_kindsEc4") : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %3 = fir.alloca i64 <{bindc_name = "c8", uniq_name = "_QFquery_with_kindsEc8"}>
  %4:2 = hlfir.declare %3 uniq_name("_QFquery_with_kindsEc8") : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
  %5 = fir.address_of(@_QFquery_with_kindsEev) : !fir.ref<!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>>
  %6:2 = hlfir.declare %5 uniq_name("_QFquery_with_kindsEev") : (!fir.ref<!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>>, !fir.ref<!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>>)
  %7 = fir.address_of(@_QMiso_fortran_env_implECint16) : !fir.ref<i32>
  %8:2 = hlfir.declare %7 uniq_name("_QMiso_fortran_env_implECint16") fortran_attrs<parameter> : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %9 = fir.address_of(@_QMiso_fortran_env_implECint32) : !fir.ref<i32>
  %10:2 = hlfir.declare %9 uniq_name("_QMiso_fortran_env_implECint32") fortran_attrs<parameter> : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %11 = fir.address_of(@_QMiso_fortran_env_implECint64) : !fir.ref<i32>
  %12:2 = hlfir.declare %11 uniq_name("_QMiso_fortran_env_implECint64") fortran_attrs<parameter> : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %13 = fir.alloca i16 <{bindc_name = "s2", uniq_name = "_QFquery_with_kindsEs2"}>
  %14:2 = hlfir.declare %13 uniq_name("_QFquery_with_kindsEs2") : (!fir.ref<i16>) -> (!fir.ref<i16>, !fir.ref<i16>)
  %15 = fir.alloca i32 <{bindc_name = "s4", uniq_name = "_QFquery_with_kindsEs4"}>
  %16:2 = hlfir.declare %15 uniq_name("_QFquery_with_kindsEs4") : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %17 = fir.alloca i64 <{bindc_name = "s8", uniq_name = "_QFquery_with_kindsEs8"}>
  %18:2 = hlfir.declare %17 uniq_name("_QFquery_with_kindsEs8") : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
  %19 = fir.load %6#0 : !fir.ref<!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>>
  %20 = fir.box_addr %19 : (!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>) -> !fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>
  %21 = mif.event_query %20 : (!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>) -> i32
  fir.store %21 to %2#0 : !fir.ref<i32>
  %22 = fir.load %6#0 : !fir.ref<!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>>
  %23 = fir.box_addr %22 : (!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>) -> !fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>
  %24 = mif.event_query %23 : (!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>) -> i64
  fir.store %24 to %4#0 : !fir.ref<i64>
  %25 = fir.load %6#0 : !fir.ref<!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>>
  %26 = fir.box_addr %25 : (!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>) -> !fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>
  %27 = mif.event_query %26 stat %14#0 : (!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, !fir.ref<i16>) -> i32
  fir.store %27 to %2#0 : !fir.ref<i32>
  %28 = fir.load %6#0 : !fir.ref<!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>>
  %29 = fir.box_addr %28 : (!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>) -> !fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>
  %30 = mif.event_query %29 stat %16#0 : (!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, !fir.ref<i32>) -> i32
  fir.store %30 to %2#0 : !fir.ref<i32>
  %31 = fir.load %6#0 : !fir.ref<!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>>
  %32 = fir.box_addr %31 : (!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>) -> !fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>
  %33 = mif.event_query %32 stat %18#0 : (!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, !fir.ref<i64>) -> i64
  fir.store %33 to %4#0 : !fir.ref<i64>
  return
}
fir.global linkonce_odr @_QMiso_fortran_envECcharacter_kinds(dense<[1, 2, 4]> : tensor<3xi32>) <{alignment = 64 : i64}> constant : !fir.array<3xi32>
fir.global internal @_QFquery_baseEev : !fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1> {
  %0 = fir.zero_bits !fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>
  %1 = fir.embox %0 : (!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>) -> !fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>
  fir.has_value %1 : !fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>
}
func.func @__mif_save_coarrays_allocate() {
  %0 = fir.alloca !fir.array<0xi64>
  %1 = fir.alloca !fir.array<1xi64>
  %2 = fir.alloca !fir.array<0xi64>
  %3 = fir.alloca !fir.array<1xi64>
  %4 = fir.alloca !fir.array<0xi64>
  %5 = fir.alloca !fir.array<1xi64>
  %6 = mif.init -> i32
  %7 = fir.address_of(@_QFquery_baseEev) : !fir.ref<!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>>
  %c1_i64 = arith.constant 1 : i64
  %c1_i64_0 = arith.constant 1 : i64
  %c0 = arith.constant 0 : index
  %8 = fir.coordinate_of %5, %c0 : (!fir.ref<!fir.array<1xi64>>, index) -> !fir.ref<i64>
  fir.store %c1_i64_0 to %8 : !fir.ref<i64>
  %9 = fir.embox %5 : (!fir.ref<!fir.array<1xi64>>) -> !fir.box<!fir.array<1xi64>>
  %c1_i64_1 = arith.constant 1 : i64
  %10 = fir.embox %4 : (!fir.ref<!fir.array<0xi64>>) -> !fir.box<!fir.array<0xi64>>
  mif.alloc_coarray %7 lcobounds %9 ucobounds %10 uniq_name("_QFquery_baseEev") : (!fir.ref<!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>>, !fir.box<!fir.array<1xi64>>, !fir.box<!fir.array<0xi64>>) -> ()
  %11 = fir.address_of(@_QFquery_with_statEev) : !fir.ref<!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>>
  %c1_i64_2 = arith.constant 1 : i64
  %c1_i64_3 = arith.constant 1 : i64
  %c0_4 = arith.constant 0 : index
  %12 = fir.coordinate_of %3, %c0_4 : (!fir.ref<!fir.array<1xi64>>, index) -> !fir.ref<i64>
  fir.store %c1_i64_3 to %12 : !fir.ref<i64>
  %13 = fir.embox %3 : (!fir.ref<!fir.array<1xi64>>) -> !fir.box<!fir.array<1xi64>>
  %c1_i64_5 = arith.constant 1 : i64
  %14 = fir.embox %2 : (!fir.ref<!fir.array<0xi64>>) -> !fir.box<!fir.array<0xi64>>
  mif.alloc_coarray %11 lcobounds %13 ucobounds %14 uniq_name("_QFquery_with_statEev") : (!fir.ref<!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>>, !fir.box<!fir.array<1xi64>>, !fir.box<!fir.array<0xi64>>) -> ()
  %15 = fir.address_of(@_QFquery_with_kindsEev) : !fir.ref<!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>>
  %c1_i64_6 = arith.constant 1 : i64
  %c1_i64_7 = arith.constant 1 : i64
  %c0_8 = arith.constant 0 : index
  %16 = fir.coordinate_of %1, %c0_8 : (!fir.ref<!fir.array<1xi64>>, index) -> !fir.ref<i64>
  fir.store %c1_i64_7 to %16 : !fir.ref<i64>
  %17 = fir.embox %1 : (!fir.ref<!fir.array<1xi64>>) -> !fir.box<!fir.array<1xi64>>
  %c1_i64_9 = arith.constant 1 : i64
  %18 = fir.embox %0 : (!fir.ref<!fir.array<0xi64>>) -> !fir.box<!fir.array<0xi64>>
  mif.alloc_coarray %15 lcobounds %17 ucobounds %18 uniq_name("_QFquery_with_kindsEev") : (!fir.ref<!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>>, !fir.box<!fir.array<1xi64>>, !fir.box<!fir.array<0xi64>>) -> ()
  return
}

// CHECK-LABEL: func.func @_QPquery_base() {
// CHECK: %[[VAL_0:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
// CHECK: %[[VAL_1:.*]] = fir.alloca i64
// CHECK: %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
// CHECK: %[[VAL_3:.*]] = fir.alloca i32 <{bindc_name = "count", uniq_name = "_QFquery_baseEcount"}>
// CHECK: %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_3]] uniq_name("_QFquery_baseEcount") : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK: %[[VAL_5:.*]] = fir.address_of(@_QFquery_baseEev) : {{.*}}
// CHECK: %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_5]] uniq_name("_QFquery_baseEev") : {{.*}}
// CHECK: %[[VAL_7:.*]] = fir.load %[[VAL_6]]#0 : {{.*}}
// CHECK: %[[VAL_8:.*]] = fir.box_addr %[[VAL_7]] : ({{.*}}) -> !fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>
// CHECK: %[[VAL_9:.*]] = fir.absent !fir.ref<i32>
// CHECK: %[[VAL_10:.*]] = fir.convert %[[VAL_8]] : (!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>) -> i64
// CHECK: %[[VAL_11:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
// CHECK: %[[VAL_12:.*]] = fir.coordinate_of %[[VAL_0]], __address : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>) -> !fir.ref<i64>
// CHECK: fir.store %[[VAL_10]] to %[[VAL_12]] : !fir.ref<i64>
// CHECK: fir.call @_QMprifPprif_event_query(%[[VAL_0]], %[[VAL_1]], %[[VAL_9]]) : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.ref<i64>, !fir.ref<i32>) -> ()
// CHECK: %[[VAL_13:.*]] = fir.load %[[VAL_1]] : !fir.ref<i64>
// CHECK: %[[VAL_14:.*]] = fir.convert %[[VAL_13]] : (i64) -> i32
// CHECK: fir.store %[[VAL_14]] to %[[VAL_4]]#0 : !fir.ref<i32>
// CHECK: return
// CHECK: }

// CHECK-LABEL: func.func @_QPquery_with_stat() {
// CHECK: %[[VAL_0:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
// CHECK: %[[VAL_1:.*]] = fir.alloca i64
// CHECK: %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
// CHECK: %[[VAL_3:.*]] = fir.alloca i32 <{bindc_name = "count", uniq_name = "_QFquery_with_statEcount"}>
// CHECK: %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_3]] uniq_name("_QFquery_with_statEcount") : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK: %[[VAL_5:.*]] = fir.address_of(@_QFquery_with_statEev) : {{.*}}
// CHECK: %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_5]] uniq_name("_QFquery_with_statEev") : {{.*}}
// CHECK: %[[VAL_7:.*]] = fir.alloca i32 <{bindc_name = "stat", uniq_name = "_QFquery_with_statEstat"}>
// CHECK: %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_7]] uniq_name("_QFquery_with_statEstat") : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK: %[[VAL_9:.*]] = fir.load %[[VAL_6]]#0 : {{.*}}
// CHECK: %[[VAL_10:.*]] = fir.box_addr %[[VAL_9]] : ({{.*}}) -> !fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>
// CHECK: %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>) -> i64
// CHECK: %[[VAL_12:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
// CHECK: %[[VAL_13:.*]] = fir.coordinate_of %[[VAL_0]], __address : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>) -> !fir.ref<i64>
// CHECK: fir.store %[[VAL_11]] to %[[VAL_13]] : !fir.ref<i64>
// CHECK: fir.call @_QMprifPprif_event_query(%[[VAL_0]], %[[VAL_1]], %[[VAL_8]]#0) : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.ref<i64>, !fir.ref<i32>) -> ()
// CHECK: %[[VAL_14:.*]] = fir.load %[[VAL_1]] : !fir.ref<i64>
// CHECK: %[[VAL_15:.*]] = fir.convert %[[VAL_14]] : (i64) -> i32
// CHECK: fir.store %[[VAL_15]] to %[[VAL_4]]#0 : !fir.ref<i32>
// CHECK: return
// CHECK: }

// CHECK-LABEL: func.func @_QPquery_with_kinds() {
// CHECK: %[[CPTR0:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
// CHECK: %[[CNT0:.*]] = fir.alloca i64
// CHECK: %[[CPTR1:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
// CHECK: %[[CNT1:.*]] = fir.alloca i64
// CHECK: %[[CPTR2:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
// CHECK: %[[CNT2:.*]] = fir.alloca i64
// CHECK: %[[CPTR3:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
// CHECK: %[[CNT3:.*]] = fir.alloca i64
// CHECK: %[[CPTR4:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
// CHECK: %[[CNT4:.*]] = fir.alloca i64
// CHECK: %[[VAL_10:.*]] = fir.dummy_scope : !fir.dscope
// CHECK: %[[VAL_11:.*]] = fir.alloca i32 <{bindc_name = "c4", uniq_name = "_QFquery_with_kindsEc4"}>
// CHECK: %[[VAL_12:.*]]:2 = hlfir.declare %[[VAL_11]] uniq_name("_QFquery_with_kindsEc4") : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK: %[[VAL_13:.*]] = fir.alloca i64 <{bindc_name = "c8", uniq_name = "_QFquery_with_kindsEc8"}>
// CHECK: %[[VAL_14:.*]]:2 = hlfir.declare %[[VAL_13]] uniq_name("_QFquery_with_kindsEc8") : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
// CHECK: %[[VAL_15:.*]] = fir.address_of(@_QFquery_with_kindsEev) : {{.*}}
// CHECK: %[[VAL_16:.*]]:2 = hlfir.declare %[[VAL_15]] uniq_name("_QFquery_with_kindsEev") : {{.*}}
// CHECK: %[[VAL_23:.*]] = fir.alloca i16 <{bindc_name = "s2", uniq_name = "_QFquery_with_kindsEs2"}>
// CHECK: %[[VAL_24:.*]]:2 = hlfir.declare %[[VAL_23]] uniq_name("_QFquery_with_kindsEs2") : (!fir.ref<i16>) -> (!fir.ref<i16>, !fir.ref<i16>)
// CHECK: %[[VAL_25:.*]] = fir.alloca i32 <{bindc_name = "s4", uniq_name = "_QFquery_with_kindsEs4"}>
// CHECK: %[[VAL_26:.*]]:2 = hlfir.declare %[[VAL_25]] uniq_name("_QFquery_with_kindsEs4") : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK: %[[VAL_27:.*]] = fir.alloca i64 <{bindc_name = "s8", uniq_name = "_QFquery_with_kindsEs8"}>
// CHECK: %[[VAL_28:.*]]:2 = hlfir.declare %[[VAL_27]] uniq_name("_QFquery_with_kindsEs8") : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
// CHECK: %[[VAL_29:.*]] = fir.load %[[VAL_16]]#0 : {{.*}}
// CHECK: %[[VAL_30:.*]] = fir.box_addr %[[VAL_29]] : ({{.*}}) -> !fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>
// CHECK: %[[VAL_31:.*]] = fir.absent !fir.ref<i32>
// CHECK: %[[VAL_32:.*]] = fir.convert %[[VAL_30]] : (!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>) -> i64
// CHECK: %[[VAL_33:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
// CHECK: %[[VAL_34:.*]] = fir.coordinate_of %[[CPTR4]], __address : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>) -> !fir.ref<i64>
// CHECK: fir.store %[[VAL_32]] to %[[VAL_34]] : !fir.ref<i64>
// CHECK: fir.call @_QMprifPprif_event_query(%[[CPTR4]], %[[CNT4]], %[[VAL_31]]) : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.ref<i64>, !fir.ref<i32>) -> ()
// CHECK: %[[VAL_35:.*]] = fir.load %[[CNT4]] : !fir.ref<i64>
// CHECK: %[[VAL_36:.*]] = fir.convert %[[VAL_35]] : (i64) -> i32
// CHECK: fir.store %[[VAL_36]] to %[[VAL_12]]#0 : !fir.ref<i32>
// CHECK: %[[VAL_37:.*]] = fir.load %[[VAL_16]]#0 : {{.*}}
// CHECK: %[[VAL_38:.*]] = fir.box_addr %[[VAL_37]] : ({{.*}}) -> !fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>
// CHECK: %[[VAL_39:.*]] = fir.absent !fir.ref<i32>
// CHECK: %[[VAL_40:.*]] = fir.convert %[[VAL_38]] : (!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>) -> i64
// CHECK: %[[VAL_41:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
// CHECK: %[[VAL_42:.*]] = fir.coordinate_of %[[CPTR3]], __address : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>) -> !fir.ref<i64>
// CHECK: fir.store %[[VAL_40]] to %[[VAL_42]] : !fir.ref<i64>
// CHECK: fir.call @_QMprifPprif_event_query(%[[CPTR3]], %[[CNT3]], %[[VAL_39]]) : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.ref<i64>, !fir.ref<i32>) -> ()
// CHECK: %[[VAL_43:.*]] = fir.load %[[CNT3]] : !fir.ref<i64>
// CHECK: fir.store %[[VAL_43]] to %[[VAL_14]]#0 : !fir.ref<i64>
// CHECK: %[[VAL_44:.*]] = fir.load %[[VAL_16]]#0 : {{.*}}
// CHECK: %[[VAL_45:.*]] = fir.box_addr %[[VAL_44]] : ({{.*}}) -> !fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>
// CHECK: %[[VAL_46:.*]] = fir.convert %[[VAL_45]] : (!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>) -> i64
// CHECK: %[[VAL_47:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
// CHECK: %[[VAL_48:.*]] = fir.coordinate_of %[[CPTR2]], __address : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>) -> !fir.ref<i64>
// CHECK: fir.store %[[VAL_46]] to %[[VAL_48]] : !fir.ref<i64>
// CHECK: %[[VAL_49:.*]] = fir.convert %[[VAL_24]]#0 : (!fir.ref<i16>) -> !fir.ref<i32>
// CHECK: fir.call @_QMprifPprif_event_query(%[[CPTR2]], %[[CNT2]], %[[VAL_49]]) : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.ref<i64>, !fir.ref<i32>) -> ()
// CHECK: %[[VAL_50:.*]] = fir.load %[[CNT2]] : !fir.ref<i64>
// CHECK: %[[VAL_51:.*]] = fir.convert %[[VAL_50]] : (i64) -> i32
// CHECK: fir.store %[[VAL_51]] to %[[VAL_12]]#0 : !fir.ref<i32>
// CHECK: %[[VAL_52:.*]] = fir.load %[[VAL_16]]#0 : {{.*}}
// CHECK: %[[VAL_53:.*]] = fir.box_addr %[[VAL_52]] : ({{.*}}) -> !fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>
// CHECK: %[[VAL_54:.*]] = fir.convert %[[VAL_53]] : (!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>) -> i64
// CHECK: %[[VAL_55:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
// CHECK: %[[VAL_56:.*]] = fir.coordinate_of %[[CPTR1]], __address : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>) -> !fir.ref<i64>
// CHECK: fir.store %[[VAL_54]] to %[[VAL_56]] : !fir.ref<i64>
// CHECK: fir.call @_QMprifPprif_event_query(%[[CPTR1]], %[[CNT1]], %[[VAL_26]]#0) : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.ref<i64>, !fir.ref<i32>) -> ()
// CHECK: %[[VAL_57:.*]] = fir.load %[[CNT1]] : !fir.ref<i64>
// CHECK: %[[VAL_58:.*]] = fir.convert %[[VAL_57]] : (i64) -> i32
// CHECK: fir.store %[[VAL_58]] to %[[VAL_12]]#0 : !fir.ref<i32>
// CHECK: %[[VAL_59:.*]] = fir.load %[[VAL_16]]#0 : {{.*}}
// CHECK: %[[VAL_60:.*]] = fir.box_addr %[[VAL_59]] : ({{.*}}) -> !fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>
// CHECK: %[[VAL_61:.*]] = fir.convert %[[VAL_60]] : (!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>) -> i64
// CHECK: %[[VAL_62:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
// CHECK: %[[VAL_63:.*]] = fir.coordinate_of %[[CPTR0]], __address : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>) -> !fir.ref<i64>
// CHECK: fir.store %[[VAL_61]] to %[[VAL_63]] : !fir.ref<i64>
// CHECK: %[[VAL_64:.*]] = fir.convert %[[VAL_28]]#0 : (!fir.ref<i64>) -> !fir.ref<i32>
// CHECK: fir.call @_QMprifPprif_event_query(%[[CPTR0]], %[[CNT0]], %[[VAL_64]]) : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.ref<i64>, !fir.ref<i32>) -> ()
// CHECK: %[[VAL_65:.*]] = fir.load %[[CNT0]] : !fir.ref<i64>
// CHECK: fir.store %[[VAL_65]] to %[[VAL_14]]#0 : !fir.ref<i64>
// CHECK: return
// CHECK: }

// RUN: fir-opt --split-input-file --cuf-implicit-device-global %s | FileCheck %s

// Test that global used in device function are flagged with the correct
// attribute.

func.func @_QMdataPsetvalue() attributes {cuf.proc_attr = #cuf.cuda_proc<global>} {
  %c6_i32 = arith.constant 6 : i32
  %21 = fir.address_of(@_QQclX6995815537abaf90e86ce166af128f3a) : !fir.ref<!fir.char<1,32>>
  %22 = fir.convert %21 : (!fir.ref<!fir.char<1,32>>) -> !fir.ref<i8>
  %c14_i32 = arith.constant 14 : i32
  %23 = fir.call @_FortranAioBeginExternalListOutput(%c6_i32, %22, %c14_i32) fastmath<contract> : (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
  return
}

func.func private @_FortranAioBeginExternalListOutput(i32, !fir.ref<i8>, i32) -> !fir.ref<i8> attributes {fir.io, fir.runtime}
fir.global linkonce @_QQclX6995815537abaf90e86ce166af128f3a constant : !fir.char<1,32> {
  %0 = fir.string_lit "cuda-implicit-device-global.fir\00"(32) : !fir.char<1,32>
  fir.has_value %0 : !fir.char<1,32>
}

// CHECK-LABEL: func.func @_QMdataPsetvalue() attributes {cuf.proc_attr = #cuf.cuda_proc<global>}

// CHECK: %[[GLOBAL:.*]] = fir.address_of(@_QQcl[[SYMBOL:.*]]) : !fir.ref<!fir.char<1,32>>
// CHECK: %[[CONV:.*]] = fir.convert %[[GLOBAL]] : (!fir.ref<!fir.char<1,32>>) -> !fir.ref<i8>
// CHECK: fir.call @_FortranAioBeginExternalListOutput(%{{.*}}, %[[CONV]], %{{.*}}) fastmath<contract> : (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
// CHECK: fir.global linkonce @_QQcl[[SYMBOL]] {data_attr = #cuf.cuda<constant>} constant : !fir.char<1,32>

// -----

func.func @_QMdataPsetvalue() {
  %c6_i32 = arith.constant 6 : i32
  %21 = fir.address_of(@_QQclX6995815537abaf90e86ce166af128f3a) : !fir.ref<!fir.char<1,32>>
  %22 = fir.convert %21 : (!fir.ref<!fir.char<1,32>>) -> !fir.ref<i8>
  %c14_i32 = arith.constant 14 : i32
  %23 = fir.call @_FortranAioBeginExternalListOutput(%c6_i32, %22, %c14_i32) fastmath<contract> : (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
  return
}

func.func private @_FortranAioBeginExternalListOutput(i32, !fir.ref<i8>, i32) -> !fir.ref<i8> attributes {fir.io, fir.runtime}
fir.global linkonce @_QQclX6995815537abaf90e86ce166af128f3a constant : !fir.char<1,32> {
  %0 = fir.string_lit "cuda-implicit-device-global.fir\00"(32) : !fir.char<1,32>
  fir.has_value %0 : !fir.char<1,32>
}

// CHECK-LABEL: func.func @_QMdataPsetvalue()
// CHECK: %[[GLOBAL:.*]] = fir.address_of(@_QQcl[[SYMBOL:.*]]) : !fir.ref<!fir.char<1,32>>
// CHECK: %[[CONV:.*]] = fir.convert %[[GLOBAL]] : (!fir.ref<!fir.char<1,32>>) -> !fir.ref<i8>
// CHECK: fir.call @_FortranAioBeginExternalListOutput(%{{.*}}, %[[CONV]], %{{.*}}) fastmath<contract> : (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
// CHECK: fir.global linkonce @_QQcl[[SYMBOL]] constant : !fir.char<1,32>

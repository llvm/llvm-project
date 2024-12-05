// RUN: fir-opt --split-input-file --cuf-device-global %s | FileCheck %s

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

// CHECK-LABEL: gpu.module @cuda_device_mod
// CHECK: fir.global linkonce @_QQclX6995815537abaf90e86ce166af128f3a

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

// CHECK-LABEL: gpu.module @cuda_device_mod
// CHECK-NOT: fir.global linkonce @_QQclX6995815537abaf90e86ce166af128f3a

// -----

func.func @_QPsub1() {
  %0 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFsub1Ei"}
  %1:2 = hlfir.declare %0 {uniq_name = "_QFsub1Ei"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %c1_i32 = arith.constant 1 : i32
  %2 = fir.convert %c1_i32 : (i32) -> index
  %c100_i32 = arith.constant 100 : i32
  %3 = fir.convert %c100_i32 : (i32) -> index
  %c1 = arith.constant 1 : index
  cuf.kernel<<<*, *>>> (%arg0 : index) = (%2 : index) to (%3 : index)  step (%c1 : index) {
    %4 = fir.convert %arg0 : (index) -> i32
    fir.store %4 to %1#1 : !fir.ref<i32>
    %5 = fir.load %1#0 : !fir.ref<i32>
    %c1_i32_0 = arith.constant 1 : i32
    %6 = arith.cmpi eq, %5, %c1_i32_0 : i32
    fir.if %6 {
      %c6_i32 = arith.constant 6 : i32
      %7 = fir.address_of(@_QQclX91d13f6e74caa2f03965d7a7c6a8fdd5) : !fir.ref<!fir.char<1,10>>
      %8 = fir.convert %7 : (!fir.ref<!fir.char<1,10>>) -> !fir.ref<i8>
      %c5_i32 = arith.constant 5 : i32
      %9 = fir.call @_FortranAioBeginExternalListOutput(%c6_i32, %8, %c5_i32) fastmath<contract> : (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
      %10 = fir.address_of(@_QQclX5465737420504153534544) : !fir.ref<!fir.char<1,11>>
      %c11 = arith.constant 11 : index
      %11:2 = hlfir.declare %10 typeparams %c11 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQclX5465737420504153534544"} : (!fir.ref<!fir.char<1,11>>, index) -> (!fir.ref<!fir.char<1,11>>, !fir.ref<!fir.char<1,11>>)
      %12 = fir.convert %11#1 : (!fir.ref<!fir.char<1,11>>) -> !fir.ref<i8>
      %13 = fir.convert %c11 : (index) -> i64
      %14 = fir.call @_FortranAioOutputAscii(%9, %12, %13) fastmath<contract> : (!fir.ref<i8>, !fir.ref<i8>, i64) -> i1
      %15 = fir.call @_FortranAioEndIoStatement(%9) fastmath<contract> : (!fir.ref<i8>) -> i32
    }
    "fir.end"() : () -> ()
  }
  return
}
func.func private @_FortranAioBeginExternalListOutput(i32, !fir.ref<i8>, i32) -> !fir.ref<i8> attributes {fir.io, fir.runtime}
fir.global linkonce @_QQclX91d13f6e74caa2f03965d7a7c6a8fdd5 constant : !fir.char<1,10> {
  %0 = fir.string_lit "dummy.cuf\00"(10) : !fir.char<1,10>
  fir.has_value %0 : !fir.char<1,10>
}
func.func private @_FortranAioOutputAscii(!fir.ref<i8>, !fir.ref<i8>, i64) -> i1 attributes {fir.io, fir.runtime}
fir.global linkonce @_QQclX5465737420504153534544 constant : !fir.char<1,11> {
  %0 = fir.string_lit "Test PASSED"(11) : !fir.char<1,11>
  fir.has_value %0 : !fir.char<1,11>
}

// CHECK: fir.global linkonce @_QQclX5465737420504153534544 {data_attr = #cuf.cuda<constant>} constant : !fir.char<1,11>

// CHECK-LABEL: gpu.module @cuda_device_mod
// CHECK: fir.global linkonce @_QQclX5465737420504153534544 {data_attr = #cuf.cuda<constant>} constant

// -----

func.func @_QPsub1() attributes {cuf.proc_attr = #cuf.cuda_proc<global>} {
  %6 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFsub1Ei"}
  %7:2 = hlfir.declare %6 {uniq_name = "_QFsub1Ei"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %12 = fir.load %7#0 : !fir.ref<i32>
  %c1_i32 = arith.constant 1 : i32
  %13 = arith.cmpi eq, %12, %c1_i32 : i32
  fir.if %13 {
    %c6_i32 = arith.constant 6 : i32
    %14 = fir.address_of(@_QQclX91d13f6e74caa2f03965d7a7c6a8fdd5) : !fir.ref<!fir.char<1,10>>
    %15 = fir.convert %14 : (!fir.ref<!fir.char<1,10>>) -> !fir.ref<i8>
    %c3_i32 = arith.constant 3 : i32
    %16 = fir.call @_FortranAioBeginExternalListOutput(%c6_i32, %15, %c3_i32) fastmath<contract> : (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
    %17 = fir.address_of(@_QQclX5465737420504153534544) : !fir.ref<!fir.char<1,11>>
    %c11 = arith.constant 11 : index
    %18:2 = hlfir.declare %17 typeparams %c11 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQclX5465737420504153534544"} : (!fir.ref<!fir.char<1,11>>, index) -> (!fir.ref<!fir.char<1,11>>, !fir.ref<!fir.char<1,11>>)
    %19 = fir.convert %18#1 : (!fir.ref<!fir.char<1,11>>) -> !fir.ref<i8>
    %20 = fir.convert %c11 : (index) -> i64
    %21 = fir.call @_FortranAioOutputAscii(%16, %19, %20) fastmath<contract> : (!fir.ref<i8>, !fir.ref<i8>, i64) -> i1
    %22 = fir.call @_FortranAioEndIoStatement(%16) fastmath<contract> : (!fir.ref<i8>) -> i32
  }
  return
}
func.func private @_FortranAioBeginExternalListOutput(i32, !fir.ref<i8>, i32) -> !fir.ref<i8> attributes {fir.io, fir.runtime}
fir.global linkonce @_QQclX91d13f6e74caa2f03965d7a7c6a8fdd5 constant : !fir.char<1,10> {
  %0 = fir.string_lit "dummy.cuf\00"(10) : !fir.char<1,10>
  fir.has_value %0 : !fir.char<1,10>
}
func.func private @_FortranAioOutputAscii(!fir.ref<i8>, !fir.ref<i8>, i64) -> i1 attributes {fir.io, fir.runtime}
fir.global linkonce @_QQclX5465737420504153534544 constant : !fir.char<1,11> {
  %0 = fir.string_lit "Test PASSED"(11) : !fir.char<1,11>
  fir.has_value %0 : !fir.char<1,11>
}
func.func private @_FortranAioEndIoStatement(!fir.ref<i8>) -> i32 attributes {fir.io, fir.runtime}

// CHECK: fir.global linkonce @_QQclX5465737420504153534544 {data_attr = #cuf.cuda<constant>} constant : !fir.char<1,11>

// CHECK-LABEL: gpu.module @cuda_device_mod 
// CHECK: fir.global linkonce @_QQclX5465737420504153534544 {data_attr = #cuf.cuda<constant>} constant

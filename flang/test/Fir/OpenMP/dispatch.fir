// RUN: fir-opt %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", fir.relocation_model = 1 : i32, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.ident = "flang version 24.0.0 (https://github.com/SunilKuravinakop/llvm-project.git 665c356fed252c9f65f99e7609997822e0e0a908)", llvm.target_triple = "x86_64-unknown-linux-gnu", omp.flags = #omp.flags<openmp_device_version = 31>, omp.is_gpu = false, omp.is_target_device = false, omp.target_triples = [], omp.version = #omp.version<version = 31>} {
  // CHECK-LABEL: func.func @_QMfuncsPfoo_variant
  func.func @_QMfuncsPfoo_variant() {
    %0 = fir.dummy_scope : !fir.dscope
    %c6_i32 = arith.constant 6 : i32
    %1 = fir.address_of(@_QQclXa0e1d7990b4b8f5533e92e3c2cc80a5d) : !fir.ref<!fir.char<1,89>>
    %2 = fir.convert %1 : (!fir.ref<!fir.char<1,89>>) -> !fir.ref<i8>
    %c35_i32 = arith.constant 35 : i32
    %3 = fir.call @_FortranAioBeginExternalListOutput(%c6_i32, %2, %c35_i32) fastmath<contract> : (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
    %4 = fir.address_of(@_QQclX696E20666F6F5F76617269616E74) : !fir.ref<!fir.char<1,14>>
    %c14 = arith.constant 14 : index
    %5:2 = hlfir.declare %4 typeparams %c14 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQclX696E20666F6F5F76617269616E74"} : (!fir.ref<!fir.char<1,14>>, index) -> (!fir.ref<!fir.char<1,14>>, !fir.ref<!fir.char<1,14>>)
    %6 = fir.convert %5#0 : (!fir.ref<!fir.char<1,14>>) -> !fir.ref<i8>
    %7 = fir.convert %c14 : (index) -> i64
    %8 = fir.call @_FortranAioOutputAscii(%3, %6, %7) fastmath<contract> : (!fir.ref<i8>, !fir.ref<i8>, i64) -> i1
    %9 = fir.call @_FortranAioEndIoStatement(%3) fastmath<contract> : (!fir.ref<i8>) -> i32
    return
  }
  // CHECK-LABEL: func.func @_QMfuncsPfoo_dispatch
  func.func @_QMfuncsPfoo_dispatch() {
    %0 = fir.dummy_scope : !fir.dscope
    %c6_i32 = arith.constant 6 : i32
    %1 = fir.address_of(@_QQclXa0e1d7990b4b8f5533e92e3c2cc80a5d) : !fir.ref<!fir.char<1,89>>
    %2 = fir.convert %1 : (!fir.ref<!fir.char<1,89>>) -> !fir.ref<i8>
    %c40_i32 = arith.constant 40 : i32
    %3 = fir.call @_FortranAioBeginExternalListOutput(%c6_i32, %2, %c40_i32) fastmath<contract> : (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
    %4 = fir.address_of(@_QQclX696E20666F6F5F6469737061746368) : !fir.ref<!fir.char<1,15>>
    %c15 = arith.constant 15 : index
    %5:2 = hlfir.declare %4 typeparams %c15 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQclX696E20666F6F5F6469737061746368"} : (!fir.ref<!fir.char<1,15>>, index) -> (!fir.ref<!fir.char<1,15>>, !fir.ref<!fir.char<1,15>>)
    %6 = fir.convert %5#0 : (!fir.ref<!fir.char<1,15>>) -> !fir.ref<i8>
    %7 = fir.convert %c15 : (index) -> i64
    %8 = fir.call @_FortranAioOutputAscii(%3, %6, %7) fastmath<contract> : (!fir.ref<i8>, !fir.ref<i8>, i64) -> i1
    %9 = fir.call @_FortranAioEndIoStatement(%3) fastmath<contract> : (!fir.ref<i8>) -> i32
    return
  }
  // CHECK-LABEL: func.func @_QQmain
  func.func @_QQmain() attributes {fir.bindc_name = "DISPATCH_TEST"} {
    %0 = fir.dummy_scope : !fir.dscope
    // CHECK: fir.call @_QMfuncsPfoo_dispatch()
    fir.call @_QMfuncsPfoo_dispatch() fastmath<contract> : () -> ()
    // CHECK: omp.dispatch {
    omp.dispatch {
      // CHECK: fir.call @_QMfuncsPfoo_variant()
      fir.call @_QMfuncsPfoo_variant() fastmath<contract> : () -> ()
      // CHECK: omp.terminator
      omp.terminator
    }
    return
  }
  // CHECK-LABEL: func.func @omp_dispatch_nowait
  func.func @omp_dispatch_nowait() {
    // CHECK: omp.dispatch nowait {
    omp.dispatch nowait {
      // CHECK: fir.call @_QMfuncsPfoo_variant()
      fir.call @_QMfuncsPfoo_variant() fastmath<contract> : () -> ()
      // CHECK: omp.terminator
      omp.terminator
    }
    return
  }
  // novariants: runtime select of base/variant address, then indirect call.
  // CHECK-LABEL: func.func @omp_dispatch_novariants
  func.func @omp_dispatch_novariants(%cond : i1) {
    // CHECK: omp.dispatch novariants(%[[COND:.*]]) {
    omp.dispatch novariants(%cond) {
      // CHECK: %[[VARIANT:.*]] = fir.address_of(@_QMfuncsPfoo_variant) : () -> ()
      %variant = fir.address_of(@_QMfuncsPfoo_variant) : () -> ()
      // CHECK: %[[BASE:.*]] = fir.address_of(@_QMfuncsPfoo_dispatch) : () -> ()
      %base = fir.address_of(@_QMfuncsPfoo_dispatch) : () -> ()
      // CHECK: %[[TARGET:.*]] = arith.select %[[COND]], %[[BASE]], %[[VARIANT]] : () -> ()
      %target = arith.select %cond, %base, %variant : () -> ()
      // CHECK: fir.call %[[TARGET]]()
      fir.call %target() fastmath<contract> : () -> ()
      // CHECK: omp.terminator
      omp.terminator
    }
    return
  }
}

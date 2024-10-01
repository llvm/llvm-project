// RUN: fir-opt --add-debug-info --mlir-print-debuginfo %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>, #dlti.dl_entry<"dlti.endianness", "little">>, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"} {
  func.func @_QMexamplePmod_sub() {
    %c2 = arith.constant 2 : index
    %1 = fir.address_of(@_QMexampleEmod_arr) : !fir.ref<!fir.array<2x2xi32>>
    %2 = fircg.ext_declare %1(%c2, %c2) {uniq_name = "_QMexampleEmod_arr"} : (!fir.ref<!fir.array<2x2xi32>>, index, index) -> !fir.ref<!fir.array<2x2xi32>> loc(#loc4)
    %3 = fir.address_of(@_QMexampleFmod_subEss) : !fir.ref<i32>
    %4 = fircg.ext_declare %3 {uniq_name = "_QMexampleFmod_subEss"} : (!fir.ref<i32>) -> !fir.ref<i32> loc(#loc5)
    return
  } loc(#loc6)
  func.func @_QQmain() attributes {fir.bindc_name = "test"} {
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %1 = fir.address_of(@_QFEarr) : !fir.ref<!fir.array<3x4xi32>>
    %2 = fircg.ext_declare %1(%c3, %c4) {uniq_name = "_QFEarr"} : (!fir.ref<!fir.array<3x4xi32>>, index, index) -> !fir.ref<!fir.array<3x4xi32>> loc(#loc2)
    %3 = fir.address_of(@_QFEs) : !fir.ref<i32>
    %4 = fircg.ext_declare %3 {uniq_name = "_QFEs"} : (!fir.ref<i32>) -> !fir.ref<i32> loc(#loc3)
    return
  } loc(#loc1)
  fir.global @_QMexampleEmod_arr : !fir.array<2x2xi32> {
    %0 = fir.zero_bits !fir.array<2x2xi32>
    fir.has_value %0 : !fir.array<2x2xi32>
  } loc(#loc4)
  fir.global internal @_QMexampleFmod_subEss : i32 {
    %c2_i32 = arith.constant 2 : i32
    fir.has_value %c2_i32 : i32
  } loc(#loc5)
  fir.global internal @_QFEarr : !fir.array<3x4xi32> {
    %0 = fir.zero_bits !fir.array<3x4xi32>
    fir.has_value %0 : !fir.array<3x4xi32>
  } loc(#loc2)
  fir.global internal @_QFEs : i32 {
    %c2_i32 = arith.constant 2 : i32
    fir.has_value %c2_i32 : i32
  } loc(#loc3)
}
#loc1 = loc("test.f90":21:1)
#loc2 = loc("test.f90":22:1)
#loc3 = loc("test.f90":23:1)
#loc4 = loc("test.f90":5:1)
#loc5 = loc("test.f90":12:1)
#loc6 = loc("test.f90":10:1)

// CHECK-DAG: #[[CU:.*]] = #llvm.di_compile_unit<{{.*}}>
// CHECK-DAG: #[[MOD:.*]] = #llvm.di_module<{{.*}}scope = #[[CU]]{{.*}}name = "example"{{.*}}>
// CHECK-DAG: #[[SP:.*]] = #llvm.di_subprogram<{{.*}}name = "_QQmain"{{.*}}>
// CHECK-DAG: #[[MOD_SP:.*]] = #llvm.di_subprogram<{{.*}}name = "mod_sub"{{.*}}retainedNodes = {{.*}}>
// CHECK-DAG: #llvm.di_global_variable<scope = #[[SP]], name = "arr"{{.*}}line = 22{{.*}}>
// CHECK-DAG: #llvm.di_global_variable<scope = #[[SP]], name = "s"{{.*}}line = 23{{.*}}>
// CHECK-DAG: #llvm.di_global_variable<scope = #[[MOD_SP]], name = "ss"{{.*}}line = 12{{.*}}>
// CHECK-DAG: #llvm.di_global_variable<scope = #[[MOD]], name = "mod_arr"{{.*}}line = 5{{.*}}>

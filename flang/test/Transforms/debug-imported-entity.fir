// RUN: fir-opt --add-debug-info --mlir-print-debuginfo %s | FileCheck %s


module attributes {dlti.dl_spec = #dlti.dl_spec<>} {
  fir.global @_QMfooEv1 : i32 {
    %0 = fir.zero_bits i32
    fir.has_value %0 : i32
  }
  fir.global internal @_QFtestExyz : i32 {
    %c12_i32 = arith.constant 12 : i32
    fir.has_value %c12_i32 : i32
  } loc(#loc4)
  func.func @test() attributes {fir.bindc_name = "test"} {
    %0 = fir.address_of(@_QMfooEv1) : !fir.ref<i32>
    %1 = fircg.ext_declare %0 {uniq_name = "_QMfooEv1"} : (!fir.ref<i32>) -> !fir.ref<i32> loc(#loc1)
    %4 = fir.address_of(@_QFtestExyz) : !fir.ref<i32>
    %5 = fircg.ext_declare %4 {uniq_name = "_QFtestExyz"} : (!fir.ref<i32>) -> !fir.ref<i32> loc(#loc4)
    return
  } loc(#loc3)
}
#loc1 = loc("test.f90":2:14)
#loc2 = loc("test.f90":6:1)
#loc3 = loc("test.f90":10:1)
#loc4 = loc("test.f90":13:1)

// CHECK: #[[MOD:.+]] = #llvm.di_module<{{.*}}name = "foo"{{.*}}>
// CHECK: #[[SP_REC:.+]] = #llvm.di_subprogram<recId = distinct[[[REC_ID:[0-9]+]]]<>, isRecSelf = true{{.*}}>
// CHECK: #[[IMP_ENTITY:.+]] = #llvm.di_imported_entity<tag = DW_TAG_imported_module, scope = #[[SP_REC]], entity = #[[MOD]]{{.*}}>
// CHECK: #[[SP:.+]] = #llvm.di_subprogram<recId = distinct[[[REC_ID]]]<>{{.*}}retainedNodes = #[[IMP_ENTITY]]>
// CHECK: #llvm.di_global_variable<scope = #[[SP]], name = "xyz"{{.*}}>

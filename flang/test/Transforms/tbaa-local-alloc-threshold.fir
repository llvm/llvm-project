// Check that -local-alloc-tbaa-threshold option limits
// the attachment of TBAA tags to accesses of locally allocated entities.
// RUN: fir-opt --fir-add-alias-tags -local-alloc-tbaa-threshold=2 %s | FileCheck %s --check-prefixes=ALL,COUNT2
// RUN: fir-opt --fir-add-alias-tags -local-alloc-tbaa-threshold=1 %s | FileCheck %s --check-prefixes=ALL,COUNT1
// RUN: fir-opt --fir-add-alias-tags -local-alloc-tbaa-threshold=0 %s | FileCheck %s --check-prefixes=ALL,COUNT0

// ALL-LABEL:   func.func @_QPtest() {
// COUNT2: fir.load{{.*}}{tbaa =
// COUNT2: fir.store{{.*}}{tbaa =
// COUNT1: fir.load{{.*}}{tbaa =
// COUNT1-NOT: fir.store{{.*}}{tbaa =
// COUNT0-NOT: fir.load{{.*}}{tbaa =
// COUNT0-NOT: fir.store{{.*}}{tbaa =
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, i64 = dense<[32, 64]> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little">, llvm.data_layout = ""} {
func.func @_QPtest() {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.alloca f32 {bindc_name = "x", uniq_name = "_QFtestEx"}
  %2 = fir.declare %1 {uniq_name = "_QFtestEx"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %3 = fir.alloca f32 {bindc_name = "y", uniq_name = "_QFtestEy"}
  %4 = fir.declare %3 {uniq_name = "_QFtestEy"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %5 = fir.load %4 : !fir.ref<f32>
  fir.store %5 to %2 : !fir.ref<f32>
  return
}
}

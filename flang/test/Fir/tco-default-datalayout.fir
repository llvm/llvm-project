// Test that tco tool sets a target independent data layout when none is
// provided. LLVM default data layout aligns i64 with 32 bits.
// RUN: tco -emit-fir %s | FileCheck %s

module {
}
// CHECK: module attributes {
// CHECK-SAME: dlti.dl_spec = #dlti.dl_spec<
// ...
// CHECK-SAME:    i64 = dense<[32, 64]> : vector<2xi64>,
// ...
// CHECK-SAME:    llvm.data_layout = ""

// Test that tco tool preserves incoming llvm.data_layout and creates a
// related dlti.dl_spec attribute. This tests a weird datalayout where
// i64 would be 128 bit aligned.
// RUN: tco -emit-fir %s | FileCheck %s

module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:128-i128:128-f80:128-n8:16:32:64-S128"} {
}
// CHECK: module attributes {
// CHECK-SAME: dlti.dl_spec = #dlti.dl_spec<
// ...
// CHECK-SAME:    #dlti.dl_entry<i64, dense<128> : vector<2xi64>>,
// ...
// CHECK-SAME:    llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:128-i128:128-f80:128-n8:16:32:64-S128"

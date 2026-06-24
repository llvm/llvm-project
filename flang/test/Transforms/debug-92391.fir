// RUN: fir-opt --add-debug-info --mlir-print-debuginfo %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>, #dlti.dl_entry<"dlti.endianness", "little">>, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"} {
  func.func @my_square_(%arg0: !fir.ref<i32> {fir.bindc_name = "x"} ) -> i32 attributes {fir.internal_name = "_QPmy_square"} {
    %0 = fir.undefined !fir.dscope
    %1 = fir.alloca i32 {bindc_name = "my_square", uniq_name = "_QFmy_squareEmy_square"}
    %2 = fircg.ext_declare %1 {uniq_name = "_QFmy_squareEmy_square"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %3 = fircg.ext_declare %arg0 dummy_scope %0 {uniq_name = "_QFmy_squareEx"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
    %4 = fir.load %3 : !fir.ref<i32>
    %5 = arith.muli %4, %4 : i32
    fir.store %5 to %2 : !fir.ref<i32>
    %6 = fir.load %2 : !fir.ref<i32>
    return %6 : i32
  } loc(#loc1)
}
#loc1 = loc("92391.f90":15:1)

// CHECK: #di_subprogram = #llvm.di_subprogram<{{.*}}name = "my_square", linkageName = "my_square_"{{.*}}>

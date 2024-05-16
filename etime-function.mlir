module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  func.func @_QPetime_test(%arg0: !fir.ref<!fir.array<2xf32>> {fir.bindc_name = "values"}, %arg1: !fir.ref<f32> {fir.bindc_name = "time"}) {
    %c9_i32 = arith.constant 9 : i32
    %c2 = arith.constant 2 : index
    %0 = fir.alloca f32
    %1 = fir.declare %arg1 {uniq_name = "_QFetime_testEtime"} : (!fir.ref<f32>) -> !fir.ref<f32>
    %2 = fir.shape %c2 : (index) -> !fir.shape<1>
    %3 = fir.declare %arg0(%2) {uniq_name = "_QFetime_testEvalues"} : (!fir.ref<!fir.array<2xf32>>, !fir.shape<1>) -> !fir.ref<!fir.array<2xf32>>
    %4 = fir.embox %3(%2) : (!fir.ref<!fir.array<2xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<2xf32>>
    %5 = fir.embox %0 : (!fir.ref<f32>) -> !fir.box<f32>
    %6 = fir.address_of(@_QQclX116781708dcf8f012d7ec1e40d743d97) : !fir.ref<!fir.char<1,71>>
    %7 = fir.convert %4 : (!fir.box<!fir.array<2xf32>>) -> !fir.box<none>
    %8 = fir.convert %5 : (!fir.box<f32>) -> !fir.box<none>
    %9 = fir.convert %6 : (!fir.ref<!fir.char<1,71>>) -> !fir.ref<i8>
    %10 = fir.call @_FortranAEtime(%7, %8, %9, %c9_i32) fastmath<contract> : (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> none
    %11 = fir.load %0 : !fir.ref<f32>
    fir.store %11 to %1 : !fir.ref<f32>
    return
  }
  func.func private @_FortranAEtime(!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> none attributes {fir.runtime}
  fir.global linkonce @_QQclX116781708dcf8f012d7ec1e40d743d97 constant : !fir.char<1,71> {
    %0 = fir.string_lit "/home/jump/llvm-project/flang/test/Lower/Intrinsics/etime-function.f90\00"(71) : !fir.char<1,71>
    fir.has_value %0 : !fir.char<1,71>
  }
}

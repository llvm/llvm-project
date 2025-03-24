// For testing:
// 1. parsing/printing (roundtripping): `fir-opt do_loop_with_local_and_local_init.mlir -o roundtrip.mlir`
// 2. Lowering locality specs during CFG: `fir-opt --cfg-conversion do_loop_with_local_and_local_init.mlir -o after_cfg_lowering.mlir`

// TODO I will add both of the above steps as proper tests when the PoC is complete.
module attributes {dlti.dl_spec = #dlti.dl_spec<i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, !llvm.ptr<270> = dense<32> : vector<4xi64>, f64 = dense<64> : vector<2xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, "dlti.endianness" = "little", "dlti.stack_alignment" = 128 : i64>, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.ident = "flang version 21.0.0 (/home/kaergawy/git/aomp20.0/llvm-project/flang c8cf5a644886bb8dd3ad19be6e3b916ffcbd222c)", llvm.target_triple = "x86_64-unknown-linux-gnu"} {

  omp.private {type = private} @local_privatizer : i32

  omp.private {type = firstprivate} @local_init_privatizer : i32 copy {
  ^bb0(%arg0: !fir.ref<i32>, %arg1: !fir.ref<i32>):
      %0 = fir.load %arg0 : !fir.ref<i32>
      fir.store %0 to %arg1 : !fir.ref<i32>
      omp.yield(%arg1 : !fir.ref<i32>)
  }

  func.func @_QPomploop() {
    %0 = fir.alloca i32 {bindc_name = "i"}
    %1:2 = hlfir.declare %0 {uniq_name = "_QFomploopEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %2 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFomploopEi"}
    %3:2 = hlfir.declare %2 {uniq_name = "_QFomploopEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %4 = fir.alloca i32 {bindc_name = "local_init_var", uniq_name = "_QFomploopElocal_init_var"}
    %5:2 = hlfir.declare %4 {uniq_name = "_QFomploopElocal_init_var"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %6 = fir.alloca i32 {bindc_name = "local_var", uniq_name = "_QFomploopElocal_var"}
    %7:2 = hlfir.declare %6 {uniq_name = "_QFomploopElocal_var"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %c1_i32 = arith.constant 1 : i32
    %8 = fir.convert %c1_i32 : (i32) -> index
    %c10_i32 = arith.constant 10 : i32
    %9 = fir.convert %c10_i32 : (i32) -> index
    %c1 = arith.constant 1 : index
    fir.do_loop %arg0 = %8 to %9 step %c1 unordered private(@local_privatizer %7#0 -> %arg1, @local_init_privatizer %5#0 -> %arg2 : !fir.ref<i32>, !fir.ref<i32>) {
      %10 = fir.convert %arg0 : (index) -> i32
      fir.store %10 to %1#1 : !fir.ref<i32>
      %12:2 = hlfir.declare %arg1 {uniq_name = "_QFomploopElocal_var"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
      %14:2 = hlfir.declare %arg2 {uniq_name = "_QFomploopElocal_init_var"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
      %16 = fir.load %1#0 : !fir.ref<i32>
      %c5_i32 = arith.constant 5 : i32
      %17 = arith.cmpi slt, %16, %c5_i32 : i32
      fir.if %17 {
        %c42_i32 = arith.constant 42 : i32
        hlfir.assign %c42_i32 to %12#0 : i32, !fir.ref<i32>
      } else {
        %c84_i32 = arith.constant 84 : i32
        hlfir.assign %c84_i32 to %14#0 : i32, !fir.ref<i32>
      }
    }
    return
  }
}

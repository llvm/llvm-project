// Test the case when AddDebugInfo pass cannot easily compute
// position of a dummy argument in the arguments list of the function.
// RUN: fir-opt --add-debug-info --mlir-print-debuginfo %s | FileCheck %s

// Only enabled on x86_64
// REQUIRES: x86-registered-target

// CHECK: #[[$ATTR_20:.+]] = #llvm.di_local_variable<scope = #di_subprogram, name = "expected", file = #di_file, line = 3, arg = 1, type = #di_basic_type>

// 'x' is a block argument at the point of fircg.ext_declare,
// but it is not the function's entry block's argument, so
// 'arg' cannot be set currently.
// CHECK: #[[$ATTR_21:.+]] = #llvm.di_local_variable<scope = #di_subprogram, name = "x", file = #di_file, line = 2, type = #di_composite_type>

// Reference Fortran code (compiled with -frepack-arrays):
// subroutine test(expected, x)
//   integer :: x(:)
//   integer :: expected
// end subroutine test

#loc1 = loc("debug-dummy-argument.f90":1:1)
#loc4 = loc("debug-dummy-argument.f90":2:14)
module attributes {dlti.dl_spec = #dlti.dl_spec<i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, i1 = dense<8> : vector<2xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i32 = dense<32> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, !llvm.ptr<270> = dense<32> : vector<4xi64>, i8 = dense<8> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, "dlti.stack_alignment" = 128 : i64, "dlti.mangling_mode" = "e", "dlti.endianness" = "little">, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", fir.target_cpu = "x86-64", llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.ident = "flang", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  func.func @test_(%arg0: !fir.ref<i32> {fir.bindc_name = "expected"} loc("debug-dummy-argument.f90":1:1), %arg1: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "x"} loc("debug-dummy-argument.f90":1:1)) attributes {fir.internal_name = "_QPtest"} {
    %0 = fir.undefined !fir.dscope loc(#loc1)
    %1 = fircg.ext_declare %arg0 dummy_scope %0 arg 1 {uniq_name = "_QFtestEexpected"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32> loc(#loc3)
    %2 = fir.is_present %arg1 : (!fir.box<!fir.array<?xi32>>) -> i1 loc(#loc4)
    cf.cond_br %2, ^bb5(%arg1 : !fir.box<!fir.array<?xi32>>), ^bb5(%arg1 : !fir.box<!fir.array<?xi32>>) loc(#loc4)
  ^bb5(%17: !fir.box<!fir.array<?xi32>> loc("debug-dummy-argument.f90":2:14)):  // 2 preds: ^bb3, ^bb4
    %18 = fircg.ext_declare %17 dummy_scope %0 {uniq_name = "_QFtestEx"} : (!fir.box<!fir.array<?xi32>>, !fir.dscope) -> !fir.box<!fir.array<?xi32>> loc(#loc4)
    cf.br ^bb8 loc(#loc1)
  ^bb8:  // 3 preds: ^bb5
    return loc(#loc5)
  } loc(#loc1)
} loc(#loc)
#loc = loc("debug-dummy-argument.f90":0:0)
#loc3 = loc("debug-dummy-argument.f90":3:14)
#loc5 = loc("debug-dummy-argument.f90":4:1)

// RUN: fir-opt --add-debug-info --mlir-print-debuginfo %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.ident = "flang version 23.0.0 (https://github.com/llvm/llvm-project.git a092e95d2d07ac974268b5d555209027b275f4da)", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  func.func @_QPvector_magnitude(%arg0: !fir.ref<f64> {fir.bindc_name = "x"}, %arg1: !fir.ref<f64> {fir.bindc_name = "y"}, %arg2: !fir.ref<f64> {fir.bindc_name = "z"}) -> f64 attributes {fir.proc_attrs = #fir.proc_attrs<pure>} {
    %0 = fir.dummy_scope : !fir.dscope
    %1 = fir.alloca f64 {bindc_name = "magnitude", uniq_name = "_QFvector_magnitudeEmagnitude"}
    %2 = fir.declare %1 {uniq_name = "_QFvector_magnitudeEmagnitude"} : (!fir.ref<f64>) -> !fir.ref<f64>
    %3 = fir.declare %arg0 dummy_scope %0 arg 1 {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFvector_magnitudeEx"} : (!fir.ref<f64>, !fir.dscope) -> !fir.ref<f64>
    %4 = fir.declare %arg1 dummy_scope %0 arg 2 {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFvector_magnitudeEy"} : (!fir.ref<f64>, !fir.dscope) -> !fir.ref<f64>
    %5 = fir.declare %arg2 dummy_scope %0 arg 3 {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFvector_magnitudeEz"} : (!fir.ref<f64>, !fir.dscope) -> !fir.ref<f64>
    %6 = fir.load %3 : !fir.ref<f64>
    %7 = fir.load %3 : !fir.ref<f64>
    %8 = arith.mulf %6, %7 fastmath<contract> : f64
    %9 = fir.load %4 : !fir.ref<f64>
    %10 = fir.load %4 : !fir.ref<f64>
    %11 = arith.mulf %9, %10 fastmath<contract> : f64
    %12 = arith.addf %8, %11 fastmath<contract> : f64
    %13 = fir.load %5 : !fir.ref<f64>
    %14 = fir.load %5 : !fir.ref<f64>
    %15 = arith.mulf %13, %14 fastmath<contract> : f64
    %16 = arith.addf %12, %15 fastmath<contract> : f64
    %17 = math.sqrt %16 fastmath<contract> : f64
    fir.store %17 to %2 : !fir.ref<f64>
    %18 = fir.load %2 : !fir.ref<f64>
    return %18 : f64
  }
  func.func @_QPsquare(%arg0: !fir.ref<f64> {fir.bindc_name = "x"}) -> f64 attributes {fir.proc_attrs = #fir.proc_attrs<elemental, pure>} {
    %0 = fir.dummy_scope : !fir.dscope
    %1 = fir.declare %arg0 dummy_scope %0 arg 1 {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFsquareEx"} : (!fir.ref<f64>, !fir.dscope) -> !fir.ref<f64>
    %2 = fir.alloca f64 {bindc_name = "x_squared", uniq_name = "_QFsquareEx_squared"}
    %3 = fir.declare %2 {uniq_name = "_QFsquareEx_squared"} : (!fir.ref<f64>) -> !fir.ref<f64>
    %4 = fir.load %1 : !fir.ref<f64>
    %5 = fir.load %1 : !fir.ref<f64>
    %6 = arith.mulf %4, %5 fastmath<contract> : f64
    fir.store %6 to %3 : !fir.ref<f64>
    %7 = fir.load %3 : !fir.ref<f64>
    return %7 : f64
  }
  func.func @_QPfactorial(%arg0: !fir.ref<i32> {fir.bindc_name = "n"}) -> i32 attributes {fir.proc_attrs = #fir.proc_attrs<recursive>} {
    %0 = fir.alloca i32 {adapt.valuebyref}
    %c1_i32 = arith.constant 1 : i32
    %1 = fir.dummy_scope : !fir.dscope
    %2 = fir.alloca i32 {bindc_name = "fact", uniq_name = "_QFfactorialEfact"}
    %3 = fir.declare %2 {uniq_name = "_QFfactorialEfact"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %4 = fir.declare %arg0 dummy_scope %1 arg 1 {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFfactorialEn"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
    %5 = fir.load %4 : !fir.ref<i32>
    %6 = arith.cmpi sle, %5, %c1_i32 : i32
    fir.if %6 {
      fir.store %c1_i32 to %3 : !fir.ref<i32>
    } else {
      %8 = fir.load %4 : !fir.ref<i32>
      %9 = fir.load %4 : !fir.ref<i32>
      %10 = arith.subi %9, %c1_i32 : i32
      fir.store %10 to %0 : !fir.ref<i32>
      %false = arith.constant false
      %11 = fir.call @_QPfactorial(%0) fastmath<contract> : (!fir.ref<i32>) -> i32
      %12 = arith.muli %8, %11 : i32
      fir.store %12 to %3 : !fir.ref<i32>
    }
    %7 = fir.load %3 : !fir.ref<i32>
    return %7 : i32
  }
}

// CHECK: #di_subprogram = #llvm.di_subprogram<{{.*}}, name = "vector_magnitude",{{.*}}subprogramFlags = "Definition|Pure"
// CHECK: #di_subprogram1 = #llvm.di_subprogram<{{.*}}, name = "square",{{.*}}subprogramFlags = "Definition|Pure|Elemental"
// CHECK: #di_subprogram2 = #llvm.di_subprogram<{{.*}}, name = "factorial",{{.*}}subprogramFlags = "Definition|Recursive"

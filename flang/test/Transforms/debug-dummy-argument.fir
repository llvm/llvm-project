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
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c2_i32 = arith.constant 2 : i32 loc(#loc2)
    %false = arith.constant false loc(#loc2)
    %c0 = arith.constant 0 : index loc(#loc2)
    %0 = fir.undefined !fir.dscope loc(#loc1)
    %1 = fircg.ext_declare %arg0 dummy_scope %0 {uniq_name = "_QFtestEexpected"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32> loc(#loc3)
    %2 = fir.is_present %arg1 : (!fir.box<!fir.array<?xi32>>) -> i1 loc(#loc4)
    cf.cond_br %2, ^bb1, ^bb3(%c0, %false : index, i1) loc(#loc4)
  ^bb1:  // pred: ^bb0
    %3 = fir.convert %arg1 : (!fir.box<!fir.array<?xi32>>) -> !fir.box<none> loc(#loc4)
    %4 = fir.call @_FortranAIsContiguous(%3) : (!fir.box<none>) -> i1 loc(#loc4)
    cf.cond_br %4, ^bb3(%c0, %false : index, i1), ^bb2 loc(#loc4)
  ^bb2:  // pred: ^bb1
    %5:3 = fir.box_dims %arg1, %c0 : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index) loc(#loc4)
    %6 = fir.box_addr %arg1 : (!fir.box<!fir.array<?xi32>>) -> !fir.ref<!fir.array<?xi32>> loc(#loc4)
    %7 = fir.is_present %6 : (!fir.ref<!fir.array<?xi32>>) -> i1 loc(#loc4)
    cf.br ^bb3(%5#1, %7 : index, i1) loc(#loc4)
  ^bb3(%8: index loc("debug-dummy-argument.f90":2:14), %9: i1 loc("debug-dummy-argument.f90":2:14)):  // 3 preds: ^bb0, ^bb1, ^bb2
    cf.cond_br %9, ^bb4, ^bb5(%arg1 : !fir.box<!fir.array<?xi32>>) loc(#loc4)
  ^bb4:  // pred: ^bb3
    %10 = fir.allocmem !fir.array<?xi32>, %8 {bindc_name = ".repacked", uniq_name = ""} loc(#loc4)
    %11 = fircg.ext_embox %10(%8) : (!fir.heap<!fir.array<?xi32>>, index) -> !fir.box<!fir.heap<!fir.array<?xi32>>> loc(#loc4)
    %12 = fir.address_of(@_QQclXa2e9fed26218fcd52fa404284303739b) : !fir.ref<!fir.char<1,43>> loc(#loc4)
    %13 = fir.convert %11 : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.box<none> loc(#loc4)
    %14 = fir.convert %arg1 : (!fir.box<!fir.array<?xi32>>) -> !fir.box<none> loc(#loc4)
    %15 = fir.convert %12 : (!fir.ref<!fir.char<1,43>>) -> !fir.ref<i8> loc(#loc4)
    fir.call @_FortranAShallowCopyDirect(%13, %14, %15, %c2_i32) : (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> () loc(#loc4)
    %16 = fircg.ext_rebox %11 : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.box<!fir.array<?xi32>> loc(#loc4)
    cf.br ^bb5(%16 : !fir.box<!fir.array<?xi32>>) loc(#loc4)
  ^bb5(%17: !fir.box<!fir.array<?xi32>> loc("debug-dummy-argument.f90":2:14)):  // 2 preds: ^bb3, ^bb4
    %18 = fircg.ext_declare %17 dummy_scope %0 {uniq_name = "_QFtestEx"} : (!fir.box<!fir.array<?xi32>>, !fir.dscope) -> !fir.box<!fir.array<?xi32>> loc(#loc4)
    cf.cond_br %2, ^bb6, ^bb8 loc(#loc1)
  ^bb6:  // pred: ^bb5
    %19 = fir.box_addr %17 : (!fir.box<!fir.array<?xi32>>) -> !fir.heap<!fir.array<?xi32>> loc(#loc1)
    %20 = fir.box_addr %arg1 : (!fir.box<!fir.array<?xi32>>) -> !fir.heap<!fir.array<?xi32>> loc(#loc1)
    %21 = fir.convert %19 : (!fir.heap<!fir.array<?xi32>>) -> index loc(#loc1)
    %22 = fir.convert %20 : (!fir.heap<!fir.array<?xi32>>) -> index loc(#loc1)
    %23 = arith.cmpi ne, %21, %22 : index loc(#loc1)
    cf.cond_br %23, ^bb7, ^bb8 loc(#loc1)
  ^bb7:  // pred: ^bb6
    %24 = fir.address_of(@_QQclXa2e9fed26218fcd52fa404284303739b) : !fir.ref<!fir.char<1,43>> loc(#loc1)
    %25 = fir.convert %arg1 : (!fir.box<!fir.array<?xi32>>) -> !fir.box<none> loc(#loc1)
    %26 = fir.convert %17 : (!fir.box<!fir.array<?xi32>>) -> !fir.box<none> loc(#loc1)
    %27 = fir.convert %24 : (!fir.ref<!fir.char<1,43>>) -> !fir.ref<i8> loc(#loc1)
    fir.call @_FortranAShallowCopyDirect(%25, %26, %27, %c1_i32) : (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> () loc(#loc1)
    fir.freemem %19 : !fir.heap<!fir.array<?xi32>> loc(#loc1)
    cf.br ^bb8 loc(#loc1)
  ^bb8:  // 3 preds: ^bb5, ^bb6, ^bb7
    return loc(#loc5)
  } loc(#loc1)
  func.func private @_FortranAShallowCopyDirect(!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) attributes {fir.runtime} loc(#loc1)
  fir.global linkonce @_QQclXa2e9fed26218fcd52fa404284303739b constant : !fir.char<1,43> {
    %0 = fir.string_lit "debug-dummy-argument.f90\00"(43) : !fir.char<1,43> loc(#loc1)
    fir.has_value %0 : !fir.char<1,43> loc(#loc1)
  } loc(#loc1)
  func.func private @_FortranAIsContiguous(!fir.box<none>) -> i1 attributes {fir.runtime} loc(#loc4)
} loc(#loc)
#loc = loc("debug-dummy-argument.f90":0:0)
#loc2 = loc(unknown)
#loc3 = loc("debug-dummy-argument.f90":3:14)
#loc5 = loc("debug-dummy-argument.f90":4:1)

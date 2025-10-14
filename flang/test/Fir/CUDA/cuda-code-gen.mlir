// RUN: fir-opt --split-input-file --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>} {
  func.func @_QQmain() attributes {fir.bindc_name = "cufkernel_global"} {
    %c0 = arith.constant 0 : index
    %0 = fir.address_of(@_QQclX3C737464696E3E00) : !fir.ref<!fir.char<1,8>>
    %c4_i32 = arith.constant 4 : i32
    %c48 = arith.constant 48 : index
    %1 = fir.convert %c48 : (index) -> i64
    %2 = fir.convert %0 : (!fir.ref<!fir.char<1,8>>) -> !fir.ref<i8>
    %3 = fir.call @_FortranACUFAllocDescriptor(%1, %2, %c4_i32) : (i64, !fir.ref<i8>, i32) -> !fir.ref<!fir.box<none>>
    %4 = fir.convert %3 : (!fir.ref<!fir.box<none>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
    %5 = fir.zero_bits !fir.heap<!fir.array<?xi32>>
    %6 = fircg.ext_embox %5(%c0) {allocator_idx = 2 : i32} : (!fir.heap<!fir.array<?xi32>>, index) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
    fir.store %6 to %4 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
    %8 = fir.load %3 : !fir.ref<!fir.box<none>>
    return
  }

  // CHECK-LABEL: llvm.func @_QQmain()
  // CHECK-COUNT-2: llvm.call @_FortranACUFAllocDescriptor 

  fir.global linkonce @_QQclX3C737464696E3E00 constant : !fir.char<1,8> {
    %0 = fir.string_lit "<stdin>\00"(8) : !fir.char<1,8>
    fir.has_value %0 : !fir.char<1,8>
  }
  func.func private @_FortranACUFAllocDescriptor(i64, !fir.ref<i8>, i32) -> !fir.ref<!fir.box<none>> attributes {fir.runtime}
}

// -----

module attributes {dlti.dl_spec = #dlti.dl_spec<f80 = dense<128> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, i64 = dense<64> : vector<2xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<270> = dense<32> : vector<4xi64>, f128 = dense<128> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i1 = dense<8> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, "dlti.endianness" = "little", "dlti.stack_alignment" = 128 : i64>} {
  func.func @_QQmain() attributes {fir.bindc_name = "test"} {
    %c10 = arith.constant 10 : index
    %c20 = arith.constant 20 : index
    %0 = fir.address_of(@_QQclX64756D6D792E6D6C697200) : !fir.ref<!fir.char<1,11>>
    %c4 = arith.constant 4 : index
    %c200 = arith.constant 200 : index
    %1 = arith.muli %c200, %c4 : index
    %c6_i32 = arith.constant 6 : i32
    %c0_i32 = arith.constant 0 : i32
    %2 = fir.convert %1 : (index) -> i64
    %3 = fir.convert %0 : (!fir.ref<!fir.char<1,11>>) -> !fir.ref<i8>
    %4 = fir.call @_FortranACUFMemAlloc(%2, %c0_i32, %3, %c6_i32) : (i64, i32, !fir.ref<i8>, i32) -> !fir.llvm_ptr<i8>
    %5 = fir.convert %4 : (!fir.llvm_ptr<i8>) -> !fir.ref<!fir.array<10x20xi32>>
    %6 = fircg.ext_embox %5(%c10, %c20) : (!fir.ref<!fir.array<10x20xi32>>, index, index) -> !fir.box<!fir.array<10x20xi32>>
    return
  }
  fir.global linkonce @_QQclX64756D6D792E6D6C697200 constant : !fir.char<1,11> {
    %0 = fir.string_lit "dummy.mlir\00"(11) : !fir.char<1,11>
    fir.has_value %0 : !fir.char<1,11>
  }
  func.func private @_FortranACUFMemAlloc(i64, i32, !fir.ref<i8>, i32) -> !fir.llvm_ptr<i8> attributes {fir.runtime}
}

// CHECK-LABEL: llvm.func @_QQmain()
// CHECK: llvm.call @_FortranACUFMemAlloc
// CHECK: llvm.call @_FortranACUFAllocDescriptor

// -----

module attributes {dlti.dl_spec = #dlti.dl_spec<f80 = dense<128> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, i64 = dense<64> : vector<2xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<270> = dense<32> : vector<4xi64>, f128 = dense<128> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i1 = dense<8> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, "dlti.endianness" = "little", "dlti.stack_alignment" = 128 : i64>} {
  func.func @_QQmain() attributes {fir.bindc_name = "p1"} {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c16_i32 = arith.constant 16 : i32
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = fir.alloca i32 {bindc_name = "iblk", uniq_name = "_QFEiblk"}
    %1 = fir.alloca i32 {bindc_name = "ithr", uniq_name = "_QFEithr"}
    %2 = fir.address_of(@_QQclX64756D6D792E6D6C697200) : !fir.ref<!fir.char<1,11>>
    %c14_i32 = arith.constant 14 : i32
    %c72 = arith.constant 72 : index
    %3 = fir.convert %c72 : (index) -> i64
    %4 = fir.convert %2 : (!fir.ref<!fir.char<1,11>>) -> !fir.ref<i8>
    %5 = fir.call @_FortranACUFAllocDescriptor(%3, %4, %c14_i32) : (i64, !fir.ref<i8>, i32) -> !fir.ref<!fir.box<none>>
    %6 = fir.convert %5 : (!fir.ref<!fir.box<none>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>
    %7 = fir.zero_bits !fir.heap<!fir.array<?x?xf32>>
    %8 = fircg.ext_embox %7(%c0, %c0) {allocator_idx = 2 : i32} : (!fir.heap<!fir.array<?x?xf32>>, index, index) -> !fir.box<!fir.heap<!fir.array<?x?xf32>>>
    fir.store %8 to %6 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>
    %9 = fir.address_of(@_QQclX64756D6D792E6D6C697200) : !fir.ref<!fir.char<1,11>>
    %c20_i32 = arith.constant 20 : i32
    %c48 = arith.constant 48 : index
    %10 = fir.convert %c48 : (index) -> i64
    %11 = fir.convert %9 : (!fir.ref<!fir.char<1,11>>) -> !fir.ref<i8>
    %12 = fir.call @_FortranACUFAllocDescriptor(%10, %11, %c20_i32) : (i64, !fir.ref<i8>, i32) -> !fir.ref<!fir.box<none>>
    %13 = fir.convert %12 : (!fir.ref<!fir.box<none>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
    %14 = fir.zero_bits !fir.heap<!fir.array<?xf32>>
    %15 = fircg.ext_embox %14(%c0) {allocator_idx = 2 : i32} : (!fir.heap<!fir.array<?xf32>>, index) -> !fir.box<!fir.heap<!fir.array<?xf32>>>
    fir.store %15 to %13 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
    %16 = fir.convert %6 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>) -> !fir.ref<!fir.box<none>>
    %17 = fir.convert %c1 : (index) -> i64
    %18 = fir.convert %c16_i32 : (i32) -> i64
    fir.call @_FortranAAllocatableSetBounds(%16, %c0_i32, %17, %18) fastmath<contract> : (!fir.ref<!fir.box<none>>, i32, i64, i64) -> ()
    fir.call @_FortranAAllocatableSetBounds(%16, %c1_i32, %17, %18) fastmath<contract> : (!fir.ref<!fir.box<none>>, i32, i64, i64) -> ()
    %21 = fir.address_of(@_QQclX64756D6D792E6D6C697200) : !fir.ref<!fir.char<1,11>>
    %c31_i32 = arith.constant 31 : i32
    %false = arith.constant false
    %22 = fir.absent !fir.box<none>
    %c-1_i64 = arith.constant -1 : i64
    %23 = fir.convert %6 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>) -> !fir.ref<!fir.box<none>>
    %24 = fir.convert %21 : (!fir.ref<!fir.char<1,11>>) -> !fir.ref<i8>
    %25 = fir.call @_FortranACUFAllocatableAllocate(%23, %c-1_i64, %false, %22, %24, %c31_i32) : (!fir.ref<!fir.box<none>>, i64, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
    %26 = fir.convert %13 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<none>>
    fir.call @_FortranAAllocatableSetBounds(%26, %c0_i32, %17, %18) fastmath<contract> : (!fir.ref<!fir.box<none>>, i32, i64, i64) -> ()
    %28 = fir.address_of(@_QQclX64756D6D792E6D6C697200) : !fir.ref<!fir.char<1,11>>
    %c34_i32 = arith.constant 34 : i32
    %false_0 = arith.constant false
    %29 = fir.absent !fir.box<none>
    %c-1_i64_1 = arith.constant -1 : i64
    %30 = fir.convert %13 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<none>>
    %31 = fir.convert %28 : (!fir.ref<!fir.char<1,11>>) -> !fir.ref<i8>
    %32 = fir.call @_FortranACUFAllocatableAllocate(%30, %c-1_i64_1, %false_0, %29, %31, %c34_i32) : (!fir.ref<!fir.box<none>>, i64, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
    %33 = fir.load %6 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>
    %34 = fircg.ext_rebox %33 : (!fir.box<!fir.heap<!fir.array<?x?xf32>>>) -> !fir.box<!fir.array<?x?xf32>>
    return
  }
  func.func private @_FortranAAllocatableSetBounds(!fir.ref<!fir.box<none>>, i32, i64, i64) -> () attributes {fir.runtime}
  fir.global linkonce @_QQclX64756D6D792E6D6C697200 constant : !fir.char<1,11> {
    %0 = fir.string_lit "dummy.mlir\00"(11) : !fir.char<1,11>
    fir.has_value %0 : !fir.char<1,11>
  }
  func.func private @_FortranACUFAllocDescriptor(i64, !fir.ref<i8>, i32) -> !fir.ref<!fir.box<none>> attributes {fir.runtime}
  func.func private @_FortranACUFAllocatableAllocate(!fir.ref<!fir.box<none>>, i64, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32 attributes {fir.runtime}
}

// CHECK-LABEL: llvm.func @_QQmain()
// CHECK-COUNT-4: llvm.call @_FortranACUFAllocDescriptor

// -----

module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, f128 = dense<128> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, i64 = dense<64> : vector<2xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, f80 = dense<128> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, "dlti.endianness" = "little", "dlti.stack_alignment" = 128 : i64>, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", gpu.container_module, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.ident = "flang version 20.0.0 (git@github.com:clementval/llvm-project.git efc2415bcce8e8a9e73e77aa122c8aba1c1fbbd2)", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  func.func @_QQmain() {
    %c1_i32 = arith.constant 1 : i32
    %c2 = arith.constant 2 : index
    %c10 = arith.constant 10 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = fir.address_of(@_QQclX3C737464696E3E00) : !fir.ref<!fir.char<1,8>>
    %c11_i32 = arith.constant 11 : i32
    %c72 = arith.constant 72 : index
    %1 = fir.convert %c72 : (index) -> i64
    %2 = fir.convert %0 : (!fir.ref<!fir.char<1,8>>) -> !fir.ref<i8>
    %3 = fir.call @_FortranACUFAllocDescriptor(%1, %2, %c11_i32) : (i64, !fir.ref<i8>, i32) -> !fir.ref<!fir.box<none>>
    %4 = fir.convert %3 : (!fir.ref<!fir.box<none>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>
    %5 = fir.zero_bits !fir.heap<!fir.array<?x?xi32>>
    %6 = fircg.ext_embox %5(%c0, %c0) {allocator_idx = 2 : i32} : (!fir.heap<!fir.array<?x?xi32>>, index, index) -> !fir.box<!fir.heap<!fir.array<?x?xi32>>>
    fir.store %6 to %4 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>
    %7 = fir.load %4 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>
    %8 = fir.box_addr %7 : (!fir.box<!fir.heap<!fir.array<?x?xi32>>>) -> !fir.heap<!fir.array<?x?xi32>>
    %c0_0 = arith.constant 0 : index
    %9:3 = fir.box_dims %7, %c0_0 : (!fir.box<!fir.heap<!fir.array<?x?xi32>>>, index) -> (index, index, index)
    %c1_1 = arith.constant 1 : index
    %10:3 = fir.box_dims %7, %c1_1 : (!fir.box<!fir.heap<!fir.array<?x?xi32>>>, index) -> (index, index, index)
    %11 = fircg.ext_embox %8(%9#1, %10#1) origin %9#0, %10#0[%c1, %c10, %c1, %c1, %c2, %c1] : (!fir.heap<!fir.array<?x?xi32>>, index, index, index, index, index, index, index, index, index, index) -> !fir.box<!fir.array<10x2xi32>>
    return
  }
  gpu.module @cuda_device_mod {
    gpu.func @_QMassumedPglob(%arg0: !fir.box<!fir.array<?x?xi32>>) kernel {
      gpu.return
    }
  }
  fir.global linkonce @_QQclX3C737464696E3E00 constant : !fir.char<1,8> {
    %0 = fir.string_lit "<stdin>\00"(8) : !fir.char<1,8>
    fir.has_value %0 : !fir.char<1,8>
  }
  func.func private @_FortranACUFAllocDescriptor(i64, !fir.ref<i8>, i32) -> !fir.ref<!fir.box<none>> attributes {fir.runtime}
  func.func private @_FortranACUFFreeDescriptor(!fir.ref<!fir.box<none>>, !fir.ref<i8>, i32) -> () attributes {fir.runtime}
}

// CHECK-LABEL: llvm.func @_QQmain()
// CHECK-COUNT-3: llvm.call @_FortranACUFAllocDescriptor

// -----

module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, f128 = dense<128> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, i64 = dense<64> : vector<2xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, f80 = dense<128> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, "dlti.endianness" = "little", "dlti.stack_alignment" = 128 : i64>, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", gpu.container_module, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.ident = "flang version 20.0.0 (git@github.com:clementval/llvm-project.git efc2415bcce8e8a9e73e77aa122c8aba1c1fbbd2)", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  func.func @_QPouter(%arg0: !fir.ref<!fir.array<100x100xf64>> {cuf.data_attr = #cuf.cuda<device>, fir.bindc_name = "a"}) {
    %c0_i32 = arith.constant 0 : i32
    %c100 = arith.constant 100 : index
    %0 = fir.alloca tuple<!fir.box<!fir.array<100x100xf64>>>
    %1 = fir.coordinate_of %0, %c0_i32 : (!fir.ref<tuple<!fir.box<!fir.array<100x100xf64>>>>, i32) -> !fir.ref<!fir.box<!fir.array<100x100xf64>>>
    %2 = fircg.ext_embox %arg0(%c100, %c100) : (!fir.ref<!fir.array<100x100xf64>>, index, index) -> !fir.box<!fir.array<100x100xf64>>
    fir.store %2 to %1 : !fir.ref<!fir.box<!fir.array<100x100xf64>>>
    return
  }
}

// CHECK-LABEL: llvm.func @_QPouter
// CHECK: _FortranACUFAllocDescriptor

// -----

func.func @_QMm1Psub1(%arg0: !fir.box<!fir.array<?xi32>> {cuf.data_attr = #cuf.cuda<device>, fir.bindc_name = "da"}, %arg1: !fir.box<!fir.array<?xi32>> {cuf.data_attr = #cuf.cuda<device>, fir.bindc_name = "db"}, %arg2: !fir.ref<i32> {fir.bindc_name = "n"}) {
  %0 = fircg.ext_rebox %arg0 : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>>
  %1 = fircg.ext_rebox %arg1 : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>>
  return
}

// CHECK-LABEL: llvm.func @_QMm1Psub1
// CHECK-COUNT-2: _FortranACUFAllocDescriptor

// -----

fir.global common @_QPshared_static__shared_mem(dense<0> : vector<28xi8>) {alignment = 8 : i64, data_attr = #cuf.cuda<shared>} : !fir.array<28xi8>

// CHECK: llvm.mlir.global common @_QPshared_static__shared_mem(dense<0> : vector<28xi8>) {addr_space = 3 : i32, alignment = 8 : i64} : !llvm.array<28 x i8>

// -----

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>} {
  func.func @_QQmain() attributes {fir.bindc_name = "cufkernel_global"} {
    %c0 = arith.constant 0 : index
    %3 = fir.call @__tgt_acc_get_deviceptr() : () -> !fir.ref<!fir.box<none>>
    %4 = fir.convert %3 : (!fir.ref<!fir.box<none>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
    %5 = fir.load %4 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
    return
  }

  // CHECK-LABEL: llvm.func @_QQmain()
  // CHECK: llvm.call @_FortranACUFAllocDescriptor

  func.func private @__tgt_acc_get_deviceptr() -> !fir.ref<!fir.box<none>>
}

// -----

module attributes {gpu.container_module, dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>} {
  fir.global @_QMm1Eda {data_attr = #cuf.cuda<device>} : !fir.box<!fir.heap<!fir.array<?x?xf32>>> {
    %c0 = arith.constant 0 : index
    %0 = fir.zero_bits !fir.heap<!fir.array<?x?xf32>>
    %1 = fircg.ext_embox %0(%c0, %c0) {allocator_idx = 2 : i32} : (!fir.heap<!fir.array<?x?xf32>>, index, index) -> !fir.box<!fir.heap<!fir.array<?x?xf32>>>
    fir.has_value %1 : !fir.box<!fir.heap<!fir.array<?x?xf32>>>
  }
  func.func @_QQmain() attributes {fir.bindc_name = "P", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+mmx", "+sse", "+sse2", "+cx8", "+x87", "+fxsr"]>} {
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %c0_i32 = arith.constant 0 : i32
    %0 = fir.address_of(@_QMm1Eda) : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>
    %8 = fir.load %0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>
    %9 = fircg.ext_rebox %8 : (!fir.box<!fir.heap<!fir.array<?x?xf32>>>) -> !fir.box<!fir.array<?x?xf32>>
    gpu.launch_func  @cuda_device_mod::@_QMm1Psub2 blocks in (%c1, %c1, %c1) threads in (%c64, %c1, %c1)  dynamic_shared_memory_size %c0_i32 args(%9 : !fir.box<!fir.array<?x?xf32>>) {cuf.proc_attr = #cuf.cuda_proc<global>}
    return
  }
  gpu.module @cuda_device_mod {
    fir.global @_QMm1Eda {data_attr = #cuf.cuda<device>} : !fir.box<!fir.heap<!fir.array<?x?xf32>>> {
      %c0 = arith.constant 0 : index
      %0 = fir.zero_bits !fir.heap<!fir.array<?x?xf32>>
      %1 = fircg.ext_embox %0(%c0, %c0) {allocator_idx = 2 : i32} : (!fir.heap<!fir.array<?x?xf32>>, index, index) -> !fir.box<!fir.heap<!fir.array<?x?xf32>>>
      fir.has_value %1 : !fir.box<!fir.heap<!fir.array<?x?xf32>>>
    }
    gpu.func @_QMm1Psub2(%arg0: !fir.box<!fir.array<?x?xf32>>) kernel {
      gpu.return
    }
  }
}

// CHECK-LABEL: llvm.func @_QQmain()
// CHECK: llvm.call @_FortranACUFAllocDescriptor

// -----

module attributes {gpu.container_module, dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>} {
  fir.global @_QMm1Eda {data_attr = #cuf.cuda<device>} : !fir.box<!fir.heap<!fir.array<?x?xf32>>> {
    %c0 = arith.constant 0 : index
    %0 = fir.zero_bits !fir.heap<!fir.array<?x?xf32>>
    %1 = fircg.ext_embox %0(%c0, %c0) {allocator_idx = 2 : i32} : (!fir.heap<!fir.array<?x?xf32>>, index, index) -> !fir.box<!fir.heap<!fir.array<?x?xf32>>>
    fir.has_value %1 : !fir.box<!fir.heap<!fir.array<?x?xf32>>>
  }
  gpu.module @cuda_device_mod {
    fir.global @_QMm1Eda {data_attr = #cuf.cuda<device>} : !fir.box<!fir.heap<!fir.array<?x?xf32>>> {
      %c0 = arith.constant 0 : index
      %0 = fir.zero_bits !fir.heap<!fir.array<?x?xf32>>
      %1 = fircg.ext_embox %0(%c0, %c0) {allocator_idx = 2 : i32} : (!fir.heap<!fir.array<?x?xf32>>, index, index) -> !fir.box<!fir.heap<!fir.array<?x?xf32>>>
      fir.has_value %1 : !fir.box<!fir.heap<!fir.array<?x?xf32>>>
    }
    func.func @_QQxxx() {
      %0 = fir.address_of(@_QMm1Eda) : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>
      %8 = fir.load %0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>
      return
    }
  }
}

// CHECK-LABEL: llvm.func @_QQxxx()
// CHECK: llvm.alloca %{{.*}} x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK-NOT: llvm.call @_FortranACUFAllocDescriptor

// -----

module attributes {gpu.container_module, dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>} {
  gpu.module @cuda_device_mod {
    fir.global @_QMkernelsEinitial_val {data_attr = #cuf.cuda<constant>} : i32 {
      %0 = fir.zero_bits i32
      fir.has_value %0 : i32
    }
    gpu.func @_QMkernelsPassign(%arg0: !fir.ref<!fir.array<?xi32>>) kernel {
      %c-1 = arith.constant -1 : index
      %c1_i32 = arith.constant 1 : i32
      %0 = arith.constant 1 : i32
      %1 = arith.addi %0, %c1_i32 : i32
      %2 = fir.address_of(@_QMkernelsEinitial_val) : !fir.ref<i32>
      %4 = fir.load %2 : !fir.ref<i32>
      %5 = fir.convert %1 : (i32) -> i64
      %6 = fircg.ext_array_coor %arg0(%c-1)<%5> : (!fir.ref<!fir.array<?xi32>>, index, i64) -> !fir.ref<i32>
      fir.store %4 to %6 : !fir.ref<i32>
      gpu.return
    }
  }
}

// CHECK: llvm.mlir.global external @_QMkernelsEinitial_val() {addr_space = 4 : i32} : i32
// CHECK-LABEL:  gpu.func @_QMkernelsPassign
// CHECK: %[[ADDROF:.*]] = llvm.mlir.addressof @_QMkernelsEinitial_val : !llvm.ptr<4>
// CHECK: %{{.*}} = llvm.addrspacecast %[[ADDROF]] : !llvm.ptr<4> to !llvm.ptr

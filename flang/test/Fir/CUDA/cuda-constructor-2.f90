// RUN: fir-opt --split-input-file --cuf-add-constructor %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", gpu.container_module, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.ident = "flang version 20.0.0 (https://github.com/llvm/llvm-project.git cae351f3453a0a26ec8eb2ddaf773c24a29d929e)", llvm.target_triple = "x86_64-unknown-linux-gnu"} {

  fir.global @_QMmtestsEn(dense<[3, 4, 5, 6, 7]> : tensor<5xi32>) {data_attr = #cuf.cuda<device>} : !fir.array<5xi32>
  fir.global @_QMmtestsEndev {data_attr = #cuf.cuda<device>} : !fir.box<!fir.heap<!fir.array<?xi32>>> {
    %c0 = arith.constant 0 : index
    %0 = fir.zero_bits !fir.heap<!fir.array<?xi32>>
    %1 = fircg.ext_embox %0(%c0) {allocator_idx = 2 : i32} : (!fir.heap<!fir.array<?xi32>>, index) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
    fir.has_value %1 : !fir.box<!fir.heap<!fir.array<?xi32>>>
  }

  gpu.module @cuda_device_mod [#nvvm.target] {
  }
}

// CHECK: gpu.module @cuda_device_mod [#nvvm.target] 

// CHECK: llvm.func internal @__cudaFortranConstructor() {
// CHECK-DAG: %[[MODULE:.*]] = cuf.register_module @cuda_device_mod -> !llvm.ptr
// CHECK-DAG: %[[VAR_NAME:.*]] = fir.address_of(@_QQ{{.*}}) : !fir.ref<!fir.char<1,12>>
// CHECK-DAG: %[[VAR_ADDR:.*]] = fir.address_of(@_QMmtestsEn) : !fir.ref<!fir.array<5xi32>>
// CHECK-DAG: %[[MODULE2:.*]] = fir.convert %[[MODULE]] : (!llvm.ptr) -> !fir.ref<!fir.llvm_ptr<i8>>
// CHECK-DAG: %[[VAR_ADDR2:.*]] = fir.convert %[[VAR_ADDR]] : (!fir.ref<!fir.array<5xi32>>) -> !fir.ref<i8>
// CHECK-DAG: %[[VAR_NAME2:.*]] = fir.convert %[[VAR_NAME]] : (!fir.ref<!fir.char<1,12>>) -> !fir.ref<i8>
// CHECK-DAG: %[[CST:.*]] = arith.constant 20 : index
// CHECK-DAG: %[[CST2:.*]] = fir.convert %[[CST]] : (index) -> i64
// CHECK-DAG: fir.call @_FortranACUFRegisterVariable(%[[MODULE2]], %[[VAR_ADDR2]], %[[VAR_NAME2]], %[[CST2]]) : (!fir.ref<!fir.llvm_ptr<i8>>, !fir.ref<i8>, !fir.ref<i8>, i64) -> none
// CHECK-DAG: %[[BOX:.*]] = fir.address_of(@_QMmtestsEndev) : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
// CHECK-DAG: %[[BOXREF:.*]] = fir.convert %[[BOX]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<i8>
// CHECK-DAG: fir.call @_FortranACUFRegisterVariable(%[[MODULE:.*]], %[[BOXREF]], %{{.*}}, %{{.*}}) 
//

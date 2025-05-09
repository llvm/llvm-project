// RUN: fir-opt --split-input-file --cuf-gpu-convert-to-llvm %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", gpu.container_module, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.ident = "flang version 20.0.0 (https://github.com/llvm/llvm-project.git cae351f3453a0a26ec8eb2ddaf773c24a29d929e)", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  gpu.module @cuda_device_mod {
    llvm.func @_QPshared_static() {
      %c0 = arith.constant 0 : i32
      %0 = cuf.shared_memory [%c0 : i32] i32 {bindc_name = "a", uniq_name = "_QFshared_staticEa"} -> !fir.ref<i32>
      %c4 = arith.constant 4 : i32
      %1 = cuf.shared_memory [%c4 : i32] i32 {bindc_name = "b", uniq_name = "_QFshared_staticEb"} -> !fir.ref<i32>
      llvm.return
    }
    llvm.mlir.global common @_QPshared_static__shared_mem(dense<0> : vector<28xi8>) {addr_space = 3 : i32, alignment = 8 : i64} : !llvm.array<28 x i8>
  }
}

// CHECK-LABEL: llvm.func @_QPshared_static()
// CHECK: %[[ADDR0:.*]] = llvm.mlir.addressof @_QPshared_static__shared_mem : !llvm.ptr<3>
// CHECK: %[[ADDRCAST0:.*]] = llvm.addrspacecast %[[ADDR0]] : !llvm.ptr<3> to !llvm.ptr
// CHECK: %[[A:.*]] = llvm.getelementptr %[[ADDRCAST0]][%c0{{.*}}] : (!llvm.ptr, i32) -> !llvm.ptr, i8
// CHECK: %[[ADDR1:.*]] = llvm.mlir.addressof @_QPshared_static__shared_mem : !llvm.ptr<3>
// CHECK: %[[ADDRCAST1:.*]] = llvm.addrspacecast %[[ADDR1]] : !llvm.ptr<3> to !llvm.ptr
// CHECK: %[[B:.*]] = llvm.getelementptr %[[ADDRCAST1]][%c4{{.*}}] : (!llvm.ptr, i32) -> !llvm.ptr, i8

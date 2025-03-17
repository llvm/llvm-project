// RUN: fir-opt --split-input-file --cuf-compute-shared-memory %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", gpu.container_module, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.ident = "flang version 20.0.0 (https://github.com/llvm/llvm-project.git cae351f3453a0a26ec8eb2ddaf773c24a29d929e)", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  gpu.module @cuda_device_mod {
    gpu.func @_QPdynshared() kernel {
      %c-1 = arith.constant -1 : index
      %6 = cuf.shared_memory !fir.array<?xf32>, %c-1 : index {bindc_name = "r", uniq_name = "_QFdynsharedEr"} -> !fir.ref<!fir.array<?xf32>>
      %7 = fir.shape %c-1 : (index) -> !fir.shape<1>
      %8 = fir.declare %6(%7) {data_attr = #cuf.cuda<shared>, uniq_name = "_QFdynsharedEr"} : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.ref<!fir.array<?xf32>>
      gpu.return
    }
  }
}

// CHECK-LABEL: gpu.module @cuda_device_mod
// CHECK: gpu.func @_QPdynshared()
// CHECK: %{{.*}} = cuf.shared_memory[%c0{{.*}} : i32] !fir.array<?xf32>, %c-1 : index {bindc_name = "r", uniq_name = "_QFdynsharedEr"} -> !fir.ref<!fir.array<?xf32>>       
// CHECK: gpu.return
// CHECK: }
// CHECK: fir.global internal @_QPdynshared__shared_mem {alignment = 4 : i64, data_attr = #cuf.cuda<shared>} : !fir.array<0xi8>

// -----

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", gpu.container_module, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.ident = "flang version 20.0.0 (https://github.com/llvm/llvm-project.git cae351f3453a0a26ec8eb2ddaf773c24a29d929e)", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  gpu.module @cuda_device_mod {
    gpu.func @_QPshared_static() attributes {cuf.proc_attr = #cuf.cuda_proc<global>} {
      %0 = cuf.shared_memory i32 {bindc_name = "a", uniq_name = "_QFshared_staticEa"} -> !fir.ref<i32>
      %1 = fir.declare %0 {data_attr = #cuf.cuda<shared>, uniq_name = "_QFshared_staticEa"} : (!fir.ref<i32>) -> !fir.ref<i32>
      %2 = cuf.shared_memory i32 {bindc_name = "b", uniq_name = "_QFshared_staticEb"} -> !fir.ref<i32>
      %3 = fir.declare %2 {data_attr = #cuf.cuda<shared>, uniq_name = "_QFshared_staticEb"} : (!fir.ref<i32>) -> !fir.ref<i32>
      %8 = cuf.shared_memory i32 {bindc_name = "c", uniq_name = "_QFshared_staticEc"} -> !fir.ref<i32>
      %9 = fir.declare %8 {data_attr = #cuf.cuda<shared>, uniq_name = "_QFshared_staticEc"} : (!fir.ref<i32>) -> !fir.ref<i32>
      %10 = cuf.shared_memory i32 {bindc_name = "d", uniq_name = "_QFshared_staticEd"} -> !fir.ref<i32>
      %11 = fir.declare %10 {data_attr = #cuf.cuda<shared>, uniq_name = "_QFshared_staticEd"} : (!fir.ref<i32>) -> !fir.ref<i32>
      %12 = cuf.shared_memory i64 {bindc_name = "e", uniq_name = "_QFshared_staticEe"} -> !fir.ref<i64>
      %13 = fir.declare %12 {data_attr = #cuf.cuda<shared>, uniq_name = "_QFshared_staticEe"} : (!fir.ref<i64>) -> !fir.ref<i64>
      %16 = cuf.shared_memory f32 {bindc_name = "r", uniq_name = "_QFshared_staticEr"} -> !fir.ref<f32>
      %17 = fir.declare %16 {data_attr = #cuf.cuda<shared>, uniq_name = "_QFshared_staticEr"} : (!fir.ref<f32>) -> !fir.ref<f32>
      gpu.return
    }
  }
}

// CHECK-LABEL: gpu.module @cuda_device_mod
// CHECK: gpu.func @_QPshared_static()
// CHECK: cuf.shared_memory[%c0{{.*}} : i32] i32 {bindc_name = "a", uniq_name = "_QFshared_staticEa"} -> !fir.ref<i32>      
// CHECK: cuf.shared_memory[%c4{{.*}} : i32] i32 {bindc_name = "b", uniq_name = "_QFshared_staticEb"} -> !fir.ref<i32>
// CHECK: cuf.shared_memory[%c8{{.*}} : i32] i32 {bindc_name = "c", uniq_name = "_QFshared_staticEc"} -> !fir.ref<i32>
// CHECK: cuf.shared_memory[%c12{{.*}} : i32] i32 {bindc_name = "d", uniq_name = "_QFshared_staticEd"} -> !fir.ref<i32>
// CHECK: cuf.shared_memory[%c16{{.*}} : i32] i64 {bindc_name = "e", uniq_name = "_QFshared_staticEe"} -> !fir.ref<i64>
// CHECK: cuf.shared_memory[%c24{{.*}} : i32] f32 {bindc_name = "r", uniq_name = "_QFshared_staticEr"} -> !fir.ref<f32>
// CHECK: gpu.return
// CHECK: }
// CHECK: fir.global internal @_QPshared_static__shared_mem(dense<0> : vector<28xi8>) {alignment = 8 : i64, data_attr = #cuf.cuda<shared>} : !fir.array<28xi8>
// CHECK: }
// CHECK: }

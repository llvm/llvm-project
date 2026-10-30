// RUN: fir-opt --split-input-file --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" %s | FileCheck %s
// RUN: fir-opt --split-input-file --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu cuda-descriptor-alloc-function=custom_alloc_desc cuda-descriptor-free-function=custom_free_desc" %s | FileCheck %s --check-prefix=CUSTOM

// A managed descriptor passed to a BIND(C) device dummy is synchronized and
// freed after the call.

func.func @bindc_call(%arg0: !fir.ref<!fir.array<10xf32>>) {
  %box = fir.embox %arg0 : (!fir.ref<!fir.array<10xf32>>) -> !fir.box<!fir.array<10xf32>>
  fir.call @c_sub(%box) proc_attrs<bind_c> : (!fir.box<!fir.array<10xf32>> {cuf.data_attr = #cuf.cuda<device>}) -> ()
  return
}
func.func private @c_sub(!fir.box<!fir.array<10xf32>> {cuf.data_attr = #cuf.cuda<device>}) attributes {fir.proc_attrs = #fir.proc_attrs<bind_c>}

// CHECK-LABEL: llvm.func @bindc_call
// CHECK: %[[DESC:.*]] = llvm.call @_FortranACUFAllocDescriptor
// CHECK: llvm.call @c_sub(%[[DESC]])
// CHECK-NEXT: llvm.call @cudadevicesynchronize_() : () -> i32
// CHECK: llvm.call @_FortranACUFFreeDescriptor(%[[DESC]], %{{.*}}, %{{.*}}) : (!llvm.ptr, !llvm.ptr, i32) -> ()
// CUSTOM-LABEL: llvm.func @bindc_call
// CUSTOM: %[[DESC:.*]] = llvm.call @custom_alloc_desc
// CUSTOM: llvm.call @cudadevicesynchronize_() : () -> i32
// CUSTOM: llvm.call @custom_free_desc(%[[DESC]], %{{.*}}, %{{.*}})

// -----

// The descriptor escapes through a store, so its lifetime is unknown and it
// is not freed.

func.func @escapes(%arg0: !fir.ref<!fir.array<10xf32>>, %arg1: !fir.ref<!fir.box<!fir.array<10xf32>>>) {
  %box = fir.embox %arg0 : (!fir.ref<!fir.array<10xf32>>) -> !fir.box<!fir.array<10xf32>>
  fir.call @c_sub(%box) proc_attrs<bind_c> : (!fir.box<!fir.array<10xf32>> {cuf.data_attr = #cuf.cuda<device>}) -> ()
  fir.store %box to %arg1 : !fir.ref<!fir.box<!fir.array<10xf32>>>
  return
}
func.func private @c_sub(!fir.box<!fir.array<10xf32>> {cuf.data_attr = #cuf.cuda<device>}) attributes {fir.proc_attrs = #fir.proc_attrs<bind_c>}

// CHECK-LABEL: llvm.func @escapes
// CHECK: llvm.call @_FortranACUFAllocDescriptor
// CHECK-NOT: FreeDescriptor

// -----

// A managed descriptor passed to a kernel launch is synchronized and freed
// after the launch, which may be asynchronous.

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, gpu.container_module} {
  func.func @_QPlaunch(%arg0: !fir.box<!fir.array<?xf32>>) {
    %c0_i32 = arith.constant 0 : i32
    %c1 = arith.constant 1 : index
    %0 = fircg.ext_rebox %arg0 : (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>>
    gpu.launch_func @cuda_device_mod::@kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) dynamic_shared_memory_size %c0_i32 args(%0 : !fir.box<!fir.array<?xf32>>) {cuf.proc_attr = #cuf.cuda_proc<global>}
    return
  }
  gpu.module @cuda_device_mod {
    gpu.func @kernel(%arg0: !fir.box<!fir.array<?xf32>>) kernel {
      gpu.return
    }
  }
}

// CHECK-LABEL: llvm.func @_QPlaunch
// CHECK: %[[DESC:.*]] = llvm.call @_FortranACUFAllocDescriptor
// CHECK: gpu.launch_func
// CHECK-NEXT: llvm.call @cudadevicesynchronize_() : () -> i32
// CHECK: llvm.call @_FortranACUFFreeDescriptor(%[[DESC]], %{{.*}}, %{{.*}}) : (!llvm.ptr, !llvm.ptr, i32) -> ()

// -----

// The descriptor reaches the launch through a branch argument in another
// block, so its lifetime is not tracked and it is not freed.

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, gpu.container_module} {
  func.func @_QPbranch(%arg0: !fir.box<!fir.array<?xf32>>) {
    %c0_i32 = arith.constant 0 : i32
    %c1 = arith.constant 1 : index
    %0 = fircg.ext_rebox %arg0 : (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>>
    cf.br ^bb1(%0 : !fir.box<!fir.array<?xf32>>)
  ^bb1(%1: !fir.box<!fir.array<?xf32>>):
    gpu.launch_func @cuda_device_mod::@kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) dynamic_shared_memory_size %c0_i32 args(%1 : !fir.box<!fir.array<?xf32>>) {cuf.proc_attr = #cuf.cuda_proc<global>}
    return
  }
  gpu.module @cuda_device_mod {
    gpu.func @kernel(%arg0: !fir.box<!fir.array<?xf32>>) kernel {
      gpu.return
    }
  }
}

// CHECK-LABEL: llvm.func @_QPbranch
// CHECK: llvm.call @_FortranACUFAllocDescriptor
// CHECK-NOT: FreeDescriptor

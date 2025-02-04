// RUN: fir-opt --split-input-file --cuf-gpu-convert-to-llvm %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", gpu.container_module, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.ident = "flang version 20.0.0 (git@github.com:clementval/llvm-project.git ddcfd4d2dc17bf66cee8c3ef6284118684a2b0e6)", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  llvm.func @_QMmod1Phost_sub() {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %2 = llvm.mlir.constant(40 : i64) : i64
    %3 = llvm.mlir.constant(16 : i32) : i32
    %4 = llvm.mlir.constant(25 : i32) : i32
    %5 = llvm.mlir.constant(21 : i32) : i32
    %6 = llvm.mlir.constant(17 : i32) : i32
    %7 = llvm.mlir.constant(1 : index) : i64
    %8 = llvm.mlir.constant(27 : i32) : i32
    %9 = llvm.mlir.constant(6 : i32) : i32
    %10 = llvm.mlir.constant(1 : i32) : i32
    %11 = llvm.mlir.constant(0 : i32) : i32
    %12 = llvm.mlir.constant(10 : index) : i64
    %13 = llvm.mlir.addressof @_QQclX91d13f6e74caa2f03965d7a7c6a8fdd5 : !llvm.ptr
    %14 = llvm.call @_FortranACUFMemAlloc(%2, %11, %13, %6) : (i64, i32, !llvm.ptr, i32) -> !llvm.ptr
    %15 = llvm.mlir.constant(10 : index) : i64
    %16 = llvm.mlir.constant(1 : index) : i64
    %17 = llvm.alloca %15 x i32 : (i64) -> !llvm.ptr
    %18 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %19 = llvm.insertvalue %17, %18[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %20 = llvm.insertvalue %17, %19[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %21 = llvm.mlir.constant(0 : index) : i64
    %22 = llvm.insertvalue %21, %20[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %23 = llvm.insertvalue %15, %22[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %24 = llvm.insertvalue %16, %23[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %25 = llvm.extractvalue %24[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %26 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %27 = llvm.insertvalue %25, %26[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %28 = llvm.insertvalue %25, %27[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %29 = llvm.mlir.constant(0 : index) : i64
    %30 = llvm.insertvalue %29, %28[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %31 = llvm.mlir.constant(10 : index) : i64
    %32 = llvm.insertvalue %31, %30[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %33 = llvm.mlir.constant(1 : index) : i64
    %34 = llvm.insertvalue %33, %32[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %35 = llvm.mlir.constant(1 : index) : i64
    %36 = llvm.mlir.constant(11 : index) : i64
    %37 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb1(%35 : i64)
  ^bb1(%38: i64):  // 2 preds: ^bb0, ^bb2
    %39 = llvm.icmp "slt" %38, %36 : i64
    llvm.cond_br %39, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %40 = llvm.mlir.constant(-1 : index) : i64
    %41 = llvm.add %38, %40 : i64
    %42 = llvm.extractvalue %34[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %43 = llvm.getelementptr %42[%41] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %11, %43 : i32, !llvm.ptr
    %44 = llvm.add %38, %37 : i64
    llvm.br ^bb1(%44 : i64)
  ^bb3:  // pred: ^bb1
    %45 = llvm.call @_FortranACUFDataTransferPtrPtr(%14, %25, %2, %11, %13, %5) : (!llvm.ptr, !llvm.ptr, i64, i32, !llvm.ptr, i32) -> !llvm.struct<()>
    gpu.launch_func  @cuda_device_mod::@_QMmod1Psub1 blocks in (%7, %7, %7) threads in (%12, %7, %7) : i64 dynamic_shared_memory_size %11 args(%14 : !llvm.ptr)
    %46 = llvm.call @_FortranACUFDataTransferPtrPtr(%25, %14, %2, %10, %13, %4) : (!llvm.ptr, !llvm.ptr, i64, i32, !llvm.ptr, i32) -> !llvm.struct<()>
    %47 = llvm.call @_FortranAioBeginExternalListOutput(%9, %13, %8) : (i32, !llvm.ptr, i32) -> !llvm.ptr
    %48 = llvm.mlir.constant(9 : i32) : i32
    %49 = llvm.mlir.zero : !llvm.ptr
    %50 = llvm.getelementptr %49[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %51 = llvm.ptrtoint %50 : !llvm.ptr to i64
    %52 = llvm.mlir.undef : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    %53 = llvm.insertvalue %51, %52[1] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> 
    %54 = llvm.mlir.constant(20240719 : i32) : i32
    %55 = llvm.insertvalue %54, %53[2] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> 
    %56 = llvm.mlir.constant(1 : i32) : i32
    %57 = llvm.trunc %56 : i32 to i8
    %58 = llvm.insertvalue %57, %55[3] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> 
    %59 = llvm.trunc %48 : i32 to i8
    %60 = llvm.insertvalue %59, %58[4] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> 
    %61 = llvm.mlir.constant(0 : i32) : i32
    %62 = llvm.trunc %61 : i32 to i8
    %63 = llvm.insertvalue %62, %60[5] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> 
    %64 = llvm.mlir.constant(0 : i32) : i32
    %65 = llvm.trunc %64 : i32 to i8
    %66 = llvm.insertvalue %65, %63[6] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> 
    %67 = llvm.mlir.constant(0 : i64) : i64
    %68 = llvm.mlir.constant(1 : i64) : i64
    %69 = llvm.insertvalue %68, %66[7, 0, 0] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> 
    %70 = llvm.insertvalue %12, %69[7, 0, 1] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> 
    %71 = llvm.insertvalue %51, %70[7, 0, 2] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> 
    %72 = llvm.mul %51, %12 : i64
    %73 = llvm.insertvalue %25, %71[0] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> 
    llvm.store %73, %1 : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>, !llvm.ptr
    llvm.return
  }
  llvm.func @_QMmod1Psub1(!llvm.ptr) -> ()
  llvm.mlir.global linkonce constant @_QQclX91d13f6e74caa2f03965d7a7c6a8fdd5()  {addr_space = 0 : i32} : !llvm.array<2 x i8> {
    %0 = llvm.mlir.constant("a\00") : !llvm.array<2 x i8>
    llvm.return %0 : !llvm.array<2 x i8>
  }
  llvm.func @_FortranAioBeginExternalListOutput(i32, !llvm.ptr, i32) -> !llvm.ptr attributes {fir.io, fir.runtime, sym_visibility = "private"}
  llvm.func @_FortranACUFMemAlloc(i64, i32, !llvm.ptr, i32) -> !llvm.ptr attributes {fir.runtime, sym_visibility = "private"}
  llvm.func @_FortranACUFDataTransferPtrPtr(!llvm.ptr, !llvm.ptr, i64, i32, !llvm.ptr, i32) -> !llvm.struct<()> attributes {fir.runtime, sym_visibility = "private"}
  llvm.func @_FortranACUFMemFree(!llvm.ptr, i32, !llvm.ptr, i32) -> !llvm.struct<()> attributes {fir.runtime, sym_visibility = "private"}
  gpu.binary @cuda_device_mod  [#gpu.object<#nvvm.target, "">]
}

// CHECK-LABEL: _QMmod1Phost_sub
// CHECK: %[[STRUCT:.*]] = llvm.alloca %{{.*}} x !llvm.struct<(ptr)> : (i32) -> !llvm.ptr
// CHECK: %[[PARAMS:.*]] = llvm.alloca %{{.*}} x !llvm.ptr : (i32) -> !llvm.ptr
// CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: %[[STRUCT_PTR:.*]] = llvm.getelementptr %[[STRUCT]][%{{.*}}, {{.*}}] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<(ptr)>
// CHECK: llvm.store %{{.*}}, %[[STRUCT_PTR]] : !llvm.ptr, !llvm.ptr
// CHECK: %[[PARAM_PTR:.*]] = llvm.getelementptr %[[PARAMS]][%[[ZERO]]] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.ptr
// CHECK: llvm.store %[[STRUCT_PTR]], %[[PARAM_PTR]] : !llvm.ptr, !llvm.ptr
// CHECK: %[[KERNEL_PTR:.*]] = llvm.mlir.addressof @_QMmod1Psub1 : !llvm.ptr
// CHECK: %[[NULL:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK: llvm.call @_FortranACUFLaunchKernel(%[[KERNEL_PTR]], {{.*}}, %[[PARAMS]], %[[NULL]])

// -----

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", gpu.container_module, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.ident = "flang version 20.0.0 (git@github.com:clementval/llvm-project.git 4116c1370ff76adf1e58eb3c39d0a14721794c70)", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  llvm.func @_FortranACUFLaunchClusterKernel(!llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i32, !llvm.ptr, !llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func @_QMmod1Psub1() attributes {cuf.cluster_dims = #cuf.cluster_dims<x = 2 : i64, y = 2 : i64, z = 1 : i64>} {
    llvm.return
  }
  llvm.func @_QQmain() attributes {fir.bindc_name = "test"} {
    %0 = llvm.mlir.constant(1 : index) : i64
    %1 = llvm.mlir.constant(2 : index) : i64
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.mlir.constant(10 : index) : i64
    gpu.launch_func  @cuda_device_mod::@_QMmod1Psub1 clusters in (%1, %1, %0) blocks in (%3, %3, %0) threads in (%3, %3, %0) : i64 dynamic_shared_memory_size %2
    llvm.return
  }
  gpu.binary @cuda_device_mod  [#gpu.object<#nvvm.target, "">]
}

// CHECK-LABEL: llvm.func @_QQmain()
// CHECK: %[[KERNEL_PTR:.*]] = llvm.mlir.addressof @_QMmod1Psub1
// CHECK: llvm.call @_FortranACUFLaunchClusterKernel(%[[KERNEL_PTR]], {{.*}})

// -----

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", gpu.container_module, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.ident = "flang version 20.0.0 (git@github.com:clementval/llvm-project.git ddcfd4d2dc17bf66cee8c3ef6284118684a2b0e6)", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  llvm.func @_QMmod1Phost_sub() {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %2 = llvm.mlir.constant(40 : i64) : i64
    %3 = llvm.mlir.constant(16 : i32) : i32
    %4 = llvm.mlir.constant(25 : i32) : i32
    %5 = llvm.mlir.constant(21 : i32) : i32
    %6 = llvm.mlir.constant(17 : i32) : i32
    %7 = llvm.mlir.constant(1 : index) : i64
    %8 = llvm.mlir.constant(27 : i32) : i32
    %9 = llvm.mlir.constant(6 : i32) : i32
    %10 = llvm.mlir.constant(1 : i32) : i32
    %11 = llvm.mlir.constant(0 : i32) : i32
    %12 = llvm.mlir.constant(10 : index) : i64
    %13 = llvm.mlir.addressof @_QQclX91d13f6e74caa2f03965d7a7c6a8fdd5 : !llvm.ptr
    %14 = llvm.call @_FortranACUFMemAlloc(%2, %11, %13, %6) : (i64, i32, !llvm.ptr, i32) -> !llvm.ptr
    gpu.launch_func  @cuda_device_mod::@_QMmod1Psub1 blocks in (%7, %7, %7) threads in (%12, %7, %7) : i64 dynamic_shared_memory_size %11 args(%14 : !llvm.ptr) {cuf.proc_attr = #cuf.cuda_proc<grid_global>}
    llvm.return
  }
  llvm.func @_QMmod1Psub1(!llvm.ptr) -> ()
  llvm.mlir.global linkonce constant @_QQclX91d13f6e74caa2f03965d7a7c6a8fdd5()  {addr_space = 0 : i32} : !llvm.array<2 x i8> {
    %0 = llvm.mlir.constant("a\00") : !llvm.array<2 x i8>
    llvm.return %0 : !llvm.array<2 x i8>
  }
  llvm.func @_FortranACUFMemAlloc(i64, i32, !llvm.ptr, i32) -> !llvm.ptr attributes {fir.runtime, sym_visibility = "private"}
  llvm.func @_FortranACUFMemFree(!llvm.ptr, i32, !llvm.ptr, i32) -> !llvm.struct<()> attributes {fir.runtime, sym_visibility = "private"}
  gpu.binary @cuda_device_mod  [#gpu.object<#nvvm.target, "">]
}

// CHECK-LABEL: llvm.func @_QMmod1Phost_sub()
// CHECK: llvm.call @_FortranACUFLaunchCooperativeKernel

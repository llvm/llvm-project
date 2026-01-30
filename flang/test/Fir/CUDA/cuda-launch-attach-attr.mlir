// RUN: fir-opt --split-input-file --cuf-launch-attach-attr %s | FileCheck %s

module attributes {gpu.container_module} {
  func.func @_QQmain() attributes {fir.bindc_name = "test"} {
    %0 = arith.constant 1 : i64
    %1 = arith.constant 2 : i64
    %3 = arith.constant 10 : i64
    gpu.launch_func  @cuda_device_mod::@_QMtest_cufk_20 blocks in (%3, %3, %0) threads in (%3, %3, %0) : i64 
    gpu.launch_func  @cuda_device_mod::@_QMtest2 blocks in (%3, %3, %0) threads in (%3, %3, %0) : i64
    gpu.launch_func  @cuda_device_mod::@_QMtest_cufk_22 blocks in (%3, %3, %0) threads in (%3, %3, %0) : i64 {cuf.proc_attr = #cuf.cuda_proc<global>}
    return
  }
  gpu.binary @cuda_device_mod  [#gpu.object<#nvvm.target, "">]
}

// CHECK-LABEL: func.func @_QQmain()
// CHECK: gpu.launch_func  @cuda_device_mod::@_QMtest_cufk_20 blocks in ({{.*}}) threads in ({{.*}}) : i64 {cuf.proc_attr = #cuf.cuda_proc<global>}
// CHECK: gpu.launch_func  @cuda_device_mod::@_QMtest2 blocks in ({{.*}}) threads in ({{.*}}) : i64 {{$}}
// CHECK: gpu.launch_func  @cuda_device_mod::@_QMtest_cufk_22 blocks in ({{.*}}) threads in ({{.*}}) : i64 {cuf.proc_attr = #cuf.cuda_proc<global>}

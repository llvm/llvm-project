// RUN: fir-opt --split-input-file --cuf-convert %s | FileCheck %s


module attributes {gpu.container_module, dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>} {
  gpu.module @cuda_device_mod {
    gpu.func @_QPsub_device1() kernel {
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      gpu.return
    }
    gpu.func @_QPsub_device2(%arg0: !fir.ref<f32>) kernel {
      cf.br ^bb1(%arg0 : !fir.ref<f32>)
    ^bb1(%0: !fir.ref<f32>):  // pred: ^bb0
      %1 = fir.declare %0 {uniq_name = "_QFsub1Ei"} : (!fir.ref<f32>) -> !fir.ref<f32>
      %cst = arith.constant 2.000000e+00 : f32
      fir.store %cst to %1 : !fir.ref<f32>
      gpu.return
    }
  }

  func.func @_QQmain() attributes {fir.bindc_name = "main"} {
    %0 = fir.alloca f32
    // CHECK: %[[ALLOCA:.*]] = fir.alloca f32
    %c1 = arith.constant 1 : index
    %c11_i32 = arith.constant 11 : i32
    %c6_i32 = arith.constant 6 : i32
    %c1_i32 = arith.constant 1 : i32
    // CHECK: gpu.launch_func  @cuda_device_mod::@_QPsub_device1 blocks in (%{{.*}}, %{{.*}}, %{{.*}}) threads in (%{{.*}}, %{{.*}}, %{{.*}})  dynamic_shared_memory_size %c0{{.*}}
    cuf.kernel_launch @cuda_device_mod::@_QPsub_device1<<<%c1_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i32>>>()

    // CHECK: gpu.launch_func  @cuda_device_mod::@_QPsub_device2 blocks in (%{{.*}}, %{{.*}}, %{{.*}}) threads in (%{{.*}}, %{{.*}}, %{{.*}})  dynamic_shared_memory_size %c0{{.*}} args(%[[ALLOCA]] : !fir.ref<f32>)
    cuf.kernel_launch @cuda_device_mod::@_QPsub_device2<<<%c1_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i32>>>(%0) : (!fir.ref<f32>)
    return
  }

}

// -----

module attributes {gpu.container_module, dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>} {
  gpu.module @cuda_device_mod {
    gpu.func @_QMmod1Psub1(%arg0: !fir.ref<!fir.array<10xi32>>) kernel {
      gpu.return
    }
  }

  func.func @_QMmod1Psub1(%arg0: !fir.ref<!fir.array<10xi32>> {cuf.data_attr = #cuf.cuda<device>, fir.bindc_name = "adev"}) attributes {cuf.cluster_dims = #cuf.cluster_dims<x = 2 : i64, y = 2 : i64, z = 1 : i64>, cuf.proc_attr = #cuf.cuda_proc<global>} {
    return
  }
  func.func @_QMmod1Phost_sub() {
    %c10 = arith.constant 10 : index
    %0 = cuf.alloc !fir.array<10xi32> {bindc_name = "adev", data_attr = #cuf.cuda<device>, uniq_name = "_QMmod1Fhost_subEadev"} -> !fir.ref<!fir.array<10xi32>>
    %1 = fir.shape %c10 : (index) -> !fir.shape<1>
    %2:2 = hlfir.declare %0(%1) {data_attr = #cuf.cuda<device>, uniq_name = "_QMmod1Fhost_subEadev"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
    %c1_i32 = arith.constant 1 : i32
    %c10_i32 = arith.constant 10 : i32
    cuf.kernel_launch @_QMmod1Psub1<<<%c1_i32, %c1_i32, %c1_i32, %c10_i32, %c1_i32, %c1_i32>>>(%2#1) : (!fir.ref<!fir.array<10xi32>>)
    return
  }
}

// CHECK-LABEL: func.func @_QMmod1Phost_sub()
// CHECK: gpu.launch_func  @cuda_device_mod::@_QMmod1Psub1 clusters in (%c2{{.*}}, %c2{{.*}}, %c1{{.*}})

// -----

module attributes {gpu.container_module, dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>} {
  gpu.module @cuda_device_mod {
    gpu.func @_QMdevptrPtest(%arg0: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>) kernel {
      gpu.return
    }
  }
  fir.global @_QMdevptrEdev_ptr {data_attr = #cuf.cuda<device>} : !fir.box<!fir.ptr<!fir.array<?xf32>>> {
    %c0 = arith.constant 0 : index
    %0 = fir.zero_bits !fir.ptr<!fir.array<?xf32>>
    %1 = fir.shape %c0 : (index) -> !fir.shape<1>
    %2 = fir.embox %0(%1) {allocator_idx = 2 : i32} : (!fir.ptr<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
    fir.has_value %2 : !fir.box<!fir.ptr<!fir.array<?xf32>>>
  }
  func.func @_QMdevptrPtest(%arg0: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>> {cuf.data_attr = #cuf.cuda<device>, fir.bindc_name = "dp"}) attributes {cuf.proc_attr = #cuf.cuda_proc<global>} {
    return
  }
  func.func @_QQmain() {
    %c1_i32 = arith.constant 1 : i32
    %c4 = arith.constant 4 : index
    %0 = cuf.alloc !fir.array<4xf32> {bindc_name = "a_dev", data_attr = #cuf.cuda<device>, uniq_name = "_QFEa_dev"} -> !fir.ref<!fir.array<4xf32>>
    %1 = fir.shape %c4 : (index) -> !fir.shape<1>
    %2 = fir.declare %0(%1) {data_attr = #cuf.cuda<device>, fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFEa_dev"} : (!fir.ref<!fir.array<4xf32>>, !fir.shape<1>) -> !fir.ref<!fir.array<4xf32>>
    %3 = fir.address_of(@_QMdevptrEdev_ptr) : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
    %4 = fir.declare %3 {data_attr = #cuf.cuda<device>, fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QMdevptrEdev_ptr"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
    %5 = fir.embox %2(%1) : (!fir.ref<!fir.array<4xf32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
    fir.store %5 to %4 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
    cuf.sync_descriptor @_QMdevptrEdev_ptr
    cuf.kernel_launch @_QMdevptrPtest<<<%c1_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i32>>>(%4) : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>)
    cuf.free %2 : !fir.ref<!fir.array<4xf32>> {data_attr = #cuf.cuda<device>}
    return
  }
}

// CHECK-LABEL: func.func @_QQmain()
// CHECK: _FortranACUFSyncGlobalDescriptor
// CHECK: %[[ADDROF:.*]] = fir.address_of(@_QMdevptrEdev_ptr) : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
// CHECK: %[[CONV_ADDR:.*]] = fir.convert %[[ADDROF]] : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>) -> !fir.llvm_ptr<i8>
// CHECK: %[[DEVADDR:.*]] = fir.call @_FortranACUFGetDeviceAddress(%[[CONV_ADDR]], %{{.*}}, %{{.*}}) : (!fir.llvm_ptr<i8>, !fir.ref<i8>, i32) -> !fir.llvm_ptr<i8>
// CHECK: %[[CONV_DEVADDR:.*]] = fir.convert %[[DEVADDR]] : (!fir.llvm_ptr<i8>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
// CHECK: gpu.launch_func  @cuda_device_mod::@_QMdevptrPtest blocks in (%{{.*}}, %{{.*}}, %{{.*}}) threads in (%{{.*}}, %{{.*}}, %{{.*}})  dynamic_shared_memory_size %{{.*}} args(%[[CONV_DEVADDR]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>)

// -----

module attributes {gpu.container_module, dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>} {
  gpu.module @cuda_device_mod {
    gpu.func @_QMdevptrPtest() kernel {
      gpu.return
    }
  }
  func.func @_QMdevptrPtest() attributes {cuf.proc_attr = #cuf.cuda_proc<grid_global>} {
    return
  }
  func.func @_QQmain() {
    %c1_i32 = arith.constant 1 : i32
    cuf.kernel_launch @_QMdevptrPtest<<<%c1_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i32>>>()
    return
  }
}

// CHECK-LABEL: func.func @_QQmain()
// CHECK: gpu.launch_func  @cuda_device_mod::@_QMdevptrPtest blocks in (%{{.*}}, %{{.*}}, %{{.*}}) threads in (%{{.*}}, %{{.*}}, %{{.*}})  dynamic_shared_memory_size %{{.*}}  {cuf.proc_attr = #cuf.cuda_proc<grid_global>}

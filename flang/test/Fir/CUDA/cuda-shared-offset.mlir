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

// -----

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", gpu.container_module, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.ident = "flang version 20.0.0 (https://github.com/llvm/llvm-project.git cae351f3453a0a26ec8eb2ddaf773c24a29d929e)", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  gpu.module @cuda_device_mod {
    gpu.func @_QMmPshareddyn(%arg0: !fir.box<!fir.array<?x?xi32>> {cuf.data_attr = #cuf.cuda<device>, fir.bindc_name = "a"}, %arg1: !fir.box<!fir.array<?x?xi32>> {cuf.data_attr = #cuf.cuda<device>, fir.bindc_name = "b"}, %arg2: i32 {fir.bindc_name = "k"}) attributes {cuf.proc_attr = #cuf.cuda_proc<global>} {
      %c1_i32 = arith.constant 1 : i32
      %c2_i32 = arith.constant 2 : i32
      %c0 = arith.constant 0 : index
      %5 = fir.address_of(@_QM__fortran_builtinsE__builtin_blockdim) : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>
      %6 = fir.declare %5 {uniq_name = "_QM__fortran_builtinsE__builtin_blockdim"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>
      %15 = fir.alloca i32
      %16 = fir.declare %15 {fortran_attrs = #fir.var_attrs<value>, uniq_name = "_QMmFss1Ek"} : (!fir.ref<i32>) -> !fir.ref<i32>
      %27 = fir.coordinate_of %6, x : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>) -> !fir.ref<i32>
      %28 = fir.load %27 : !fir.ref<i32>
      %29 = fir.convert %28 : (i32) -> i64
      %30 = fir.convert %29 : (i64) -> index
      %31 = arith.cmpi sgt, %30, %c0 : index
      %32 = arith.select %31, %30, %c0 : index
      %33 = fir.coordinate_of %6, y : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>) -> !fir.ref<i32>
      %34 = fir.load %33 : !fir.ref<i32>
      %35 = fir.convert %34 : (i32) -> i64
      %36 = fir.convert %35 : (i64) -> index
      %37 = arith.cmpi sgt, %36, %c0 : index
      %38 = arith.select %37, %36, %c0 : index
      %39 = cuf.shared_memory !fir.array<?x?xi32>, %32, %38 : index, index {bindc_name = "s1", uniq_name = "_QMmFss1Es1"} -> !fir.ref<!fir.array<?x?xi32>>
      %40 = fir.shape %32, %38 : (index, index) -> !fir.shape<2>
      %41 = fir.declare %39(%40) {data_attr = #cuf.cuda<shared>, uniq_name = "_QMmFss1Es1"} : (!fir.ref<!fir.array<?x?xi32>>, !fir.shape<2>) -> !fir.ref<!fir.array<?x?xi32>>
      %42 = fir.load %16 : !fir.ref<i32>
      %43 = arith.muli %42, %c2_i32 : i32
      %44 = fir.convert %43 : (i32) -> i64
      %45 = fir.convert %44 : (i64) -> index
      %46 = arith.cmpi sgt, %45, %c0 : index
      %47 = arith.select %46, %45, %c0 : index
      %48 = fir.load %16 : !fir.ref<i32>
      %49 = fir.convert %48 : (i32) -> i64
      %50 = fir.convert %49 : (i64) -> index
      %51 = arith.cmpi sgt, %50, %c0 : index
      %52 = arith.select %51, %50, %c0 : index
      %53 = cuf.shared_memory !fir.array<?x?xi32>, %47, %52 : index, index {bindc_name = "s2", uniq_name = "_QMmFss1Es2"} -> !fir.ref<!fir.array<?x?xi32>>
      gpu.return
    }
  }
}

// CHECK: gpu.func @_QMmPshareddyn(%arg0: !fir.box<!fir.array<?x?xi32>> {cuf.data_attr = #cuf.cuda<device>, fir.bindc_name = "a"}, %arg1: !fir.box<!fir.array<?x?xi32>> {cuf.data_attr = #cuf.cuda<device>, fir.bindc_name = "b"}, %arg2: i32 {fir.bindc_name = "k"}) attributes {cuf.proc_attr = #cuf.cuda_proc<global>} {
// CHECK: %[[EXTENT0:.*]] = arith.select 
// CHECK: %[[EXTENT1:.*]] = arith.select 
// CHECK: %[[SIZE_EXTENT:.*]] = arith.muli %c4{{.*}}, %[[EXTENT0]] : index
// CHECK: %[[DYNSIZE:.*]] = arith.muli %[[SIZE_EXTENT]], %[[EXTENT1]] : index
// CHECK: cuf.shared_memory[%c0{{.*}} : i32] !fir.array<?x?xi32>, %9, %15 : index, index {bindc_name = "s1", uniq_name = "_QMmFss1Es1"} -> !fir.ref<!fir.array<?x?xi32>>
// CHECK: %[[CONV_DYNSIZE:.*]] = fir.convert %[[DYNSIZE]] : (index) -> i32
// CHECK: cuf.shared_memory[%[[CONV_DYNSIZE]] : i32] !fir.array<?x?xi32>, %26, %31 : index, index {bindc_name = "s2", uniq_name = "_QMmFss1Es2"} -> !fir.ref<!fir.array<?x?xi32>>

// -----

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", gpu.container_module, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.ident = "flang version 20.0.0 (https://github.com/llvm/llvm-project.git cae351f3453a0a26ec8eb2ddaf773c24a29d929e)", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  gpu.module @cuda_device_mod {
    gpu.func @_QPnoshared() kernel {
      gpu.return
    }
  }
}

// CHECK-LABEL: gpu.func @_QPnoshared()
// CHECK-NOT: fir.global internal @_QPnoshared__shared_mem

// -----

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", gpu.container_module, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.ident = "flang version 20.0.0 (https://github.com/llvm/llvm-project.git cae351f3453a0a26ec8eb2ddaf773c24a29d929e)", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  gpu.module @cuda_device_mod {
    gpu.func @_QMmtestsPtestany(%arg0: !fir.ref<!fir.array<?xf32>> {cuf.data_attr = #cuf.cuda<device>, fir.bindc_name = "a"}) attributes {cuf.proc_attr = #cuf.cuda_proc<global>} {
      %0 = fir.dummy_scope : !fir.dscope
      %c-1 = arith.constant -1 : index
      %1 = fir.shape %c-1 : (index) -> !fir.shape<1>
      %2:2 = hlfir.declare %arg0(%1) dummy_scope %0 {data_attr = #cuf.cuda<device>, uniq_name = "_QMmtestsFtestanyEa"} : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>)
      %3 = fir.address_of(@_QM__fortran_builtinsE__builtin_blockdim) : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>
      %4:2 = hlfir.declare %3 {uniq_name = "_QM__fortran_builtinsE__builtin_blockdim"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>) -> (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>)
      %5 = fir.address_of(@_QM__fortran_builtinsE__builtin_blockidx) : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>
      %6:2 = hlfir.declare %5 {uniq_name = "_QM__fortran_builtinsE__builtin_blockidx"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>) -> (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>)
      %c-1_0 = arith.constant -1 : index
      %7 = cuf.shared_memory !fir.array<?xf64>, %c-1_0 : index {bindc_name = "dmasks", uniq_name = "_QMmtestsFtestanyEdmasks"} -> !fir.ref<!fir.array<?xf64>>
      %8 = fir.shape %c-1_0 : (index) -> !fir.shape<1>
      %9:2 = hlfir.declare %7(%8) {data_attr = #cuf.cuda<shared>, uniq_name = "_QMmtestsFtestanyEdmasks"} : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>)
      %10 = fir.address_of(@_QM__fortran_builtinsE__builtin_griddim) : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>
      %11:2 = hlfir.declare %10 {uniq_name = "_QM__fortran_builtinsE__builtin_griddim"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>) -> (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>)
      %12 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QMmtestsFtestanyEi"}
      %13:2 = hlfir.declare %12 {uniq_name = "_QMmtestsFtestanyEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
      %14 = fir.alloca i32 {bindc_name = "iam", uniq_name = "_QMmtestsFtestanyEiam"}
      %15:2 = hlfir.declare %14 {uniq_name = "_QMmtestsFtestanyEiam"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
      %16 = fir.alloca i32 {bindc_name = "j", uniq_name = "_QMmtestsFtestanyEj"}
      %17:2 = hlfir.declare %16 {uniq_name = "_QMmtestsFtestanyEj"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
      %c-1_1 = arith.constant -1 : index
      %18 = cuf.shared_memory !fir.array<?xf32>, %c-1_1 : index {bindc_name = "smasks", uniq_name = "_QMmtestsFtestanyEsmasks"} -> !fir.ref<!fir.array<?xf32>>
      %19 = fir.shape %c-1_1 : (index) -> !fir.shape<1>
      %20:2 = hlfir.declare %18(%19) {data_attr = #cuf.cuda<shared>, uniq_name = "_QMmtestsFtestanyEsmasks"} : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>)
      gpu.return
    }
  }
}

// CHECK-LABEL: gpu.func @_QMmtestsPtestany
// CHECK: %{{.*}} = cuf.shared_memory[%c0{{.*}} : i32] !fir.array<?xf64>, %c-1{{.*}} : index {bindc_name = "dmasks", uniq_name = "_QMmtestsFtestanyEdmasks"} -> !fir.ref<!fir.array<?xf64>>
// CHECK: %{{.*}} = cuf.shared_memory[%c0{{.*}} : i32] !fir.array<?xf32>, %c-1{{.*}} : index {bindc_name = "smasks", uniq_name = "_QMmtestsFtestanyEsmasks"} -> !fir.ref<!fir.array<?xf32>>

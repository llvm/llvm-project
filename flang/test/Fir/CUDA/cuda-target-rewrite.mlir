// RUN: fir-opt --split-input-file --target-rewrite="target=x86_64-unknown-linux-gnu" %s | FileCheck %s

module attributes {fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"} {

gpu.module @testmod {
  gpu.func @_QPvcpowdk(%arg0: !fir.ref<complex<f64>> {cuf.data_attr = #cuf.cuda<device>, fir.bindc_name = "a"}) attributes {cuf.proc_attr = #cuf.cuda_proc<global>} {
    %0 = fir.alloca i64
    %1 = fir.load %0 : !fir.ref<i64>
    %2 = fir.load %arg0 : !fir.ref<complex<f64>>
    %3 = fir.call @_FortranAzpowk(%2, %1) fastmath<contract> : (complex<f64>, i64) -> complex<f64>
    gpu.return
  }
  func.func private @_FortranAzpowk(complex<f64>, i64) -> complex<f64> attributes {fir.bindc_name = "_FortranAzpowk", fir.runtime}
}

// CHECK-LABEL: gpu.func @_QPvcpowdk
// CHECK: %{{.*}} = fir.call @_FortranAzpowk(%{{.*}}, %{{.*}}, %{{.*}}) fastmath<contract> : (f64, f64, i64) -> tuple<f64, f64>
// CHECK: func.func private @_FortranAzpowk(f64, f64, i64) -> tuple<f64, f64> attributes {fir.bindc_name = "_FortranAzpowk", fir.runtime}
}

// -----

module attributes {fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"} {

gpu.module @testmod {
  gpu.func @_QPtest(%arg0: complex<f64>) -> (complex<f64>) {
    gpu.return %arg0 : complex<f64>
  }
}
}

// CHECK-LABEL: gpu.func @_QPtest
// CHECK-SAME: (%arg0: f64, %arg1: f64) -> tuple<f64, f64> {
// CHECK: gpu.return %{{.*}} : tuple<f64, f64>

// -----
module attributes {gpu.container_module, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"} {

gpu.module @testmod {
  gpu.func @_QPtest(%arg0: complex<f64>) -> () kernel {
    gpu.return
  }
}

func.func @main(%arg0: complex<f64>) {
  %0 = llvm.mlir.constant(0 : i64) : i64
  %1 = llvm.mlir.constant(0 : i32) : i32
  gpu.launch_func  @testmod::@_QPtest blocks in (%0, %0, %0) threads in (%0, %0, %0) : i64 dynamic_shared_memory_size %1 args(%arg0 : complex<f64>) {cuf.proc_attr = #cuf.cuda_proc<global>}
  return
}

}

// CHECK-LABEL: gpu.func @_QPtest
// CHECK-SAME: (%arg0: f64, %arg1: f64) kernel {
// CHECK: gpu.return
// CHECK: gpu.launch_func  @testmod::@_QPtest blocks in (%{{.*}}, %{{.*}}, %{{.*}}) threads in (%{{.*}}, %{{.*}}, %{{.*}}) : i64 dynamic_shared_memory_size %{{.*}} args(%{{.*}} : f64, %{{.*}} : f64) {cuf.proc_attr = #cuf.cuda_proc<global>}

// -----

module attributes {gpu.container_module, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  gpu.module @testmod {
    gpu.func @_QMbarPfoo(%arg0: f32, %arg1: !fir.ref<!fir.array<100xf32>>, %arg2: !fir.boxchar<1>) workgroup(%arg3 : memref<1xf32, #gpu.address_space<workgroup>> {llvm.align = 16 : i32}) {
      %c0 = arith.constant 0 : index
      memref.store %arg0, %arg3[%c0] : memref<1xf32, #gpu.address_space<workgroup>>
      gpu.return
    }
// CHECK-LABEL: gpu.func @_QMbarPfoo(
// CHECK-SAME: %{{.*}}: f32, %{{.*}}: !fir.ref<!fir.array<100xf32>>, %[[CHAR:.*]]: !fir.ref<!fir.char<1,?>>, %[[LENGTH:.*]]: i64) workgroup(%[[WORKGROUP:.*]] : memref<1xf32, #gpu.address_space<workgroup>> {llvm.align = 16 : i32}) {
// CHECK: %{{.*}} = fir.emboxchar %[[CHAR]], %[[LENGTH]] : (!fir.ref<!fir.char<1,?>>, i64) -> !fir.boxchar<1>
// CHECK: memref.store %{{.*}}, %[[WORKGROUP]][%{{.*}}] : memref<1xf32, #gpu.address_space<workgroup>>

    gpu.func @_QMbarPfoo2(%arg0: f32, %arg1: !fir.ref<!fir.array<100xf32>>, %arg2: !fir.boxchar<1>) workgroup(%arg3 : memref<1xf32, #gpu.address_space<workgroup>> {llvm.align = 16 : i32}, %arg4 : memref<1xf32, #gpu.address_space<workgroup>> {llvm.align = 16 : i32}) {
      %c0 = arith.constant 0 : index
      memref.store %arg0, %arg3[%c0] : memref<1xf32, #gpu.address_space<workgroup>>
      memref.store %arg0, %arg4[%c0] : memref<1xf32, #gpu.address_space<workgroup>>
      gpu.return
    }
// CHECK-LABEL: gpu.func @_QMbarPfoo2(
// CHECK-SAME: %{{.*}}: f32, %{{.*}}: !fir.ref<!fir.array<100xf32>>, %[[CHAR:.*]]: !fir.ref<!fir.char<1,?>>, %[[LENGTH:.*]]: i64) workgroup(%[[WG1:.*]] : memref<1xf32, #gpu.address_space<workgroup>> {llvm.align = 16 : i32}, %[[WG2:.*]] : memref<1xf32, #gpu.address_space<workgroup>> {llvm.align = 16 : i32}) {
// CHECK: %{{.*}} = fir.emboxchar %[[CHAR]], %[[LENGTH]] : (!fir.ref<!fir.char<1,?>>, i64) -> !fir.boxchar<1>
// CHECK: memref.store %{{.*}}, %[[WG1]][%{{.*}}] : memref<1xf32, #gpu.address_space<workgroup>>
// CHECK: memref.store %{{.*}}, %[[WG2]][%{{.*}}] : memref<1xf32, #gpu.address_space<workgroup>>

    gpu.func @_QMbarPprivate(%arg0: f32, %arg1: !fir.boxchar<1>) workgroup(%arg2 : memref<1xf32, #gpu.address_space<workgroup>> {llvm.align = 16 : i32}) private(%arg3 : memref<1xf32, #gpu.address_space<private>> {llvm.align = 16 : i32}) {
      %c0 = arith.constant 0 : index
      memref.store %arg0, %arg2[%c0] : memref<1xf32, #gpu.address_space<workgroup>>
      memref.store %arg0, %arg3[%c0] : memref<1xf32, #gpu.address_space<private>>
      gpu.return
    }
// CHECK-LABEL: gpu.func @_QMbarPprivate(
// CHECK-SAME: %{{.*}}: f32, %[[CHAR:.*]]: !fir.ref<!fir.char<1,?>>, %[[LENGTH:.*]]: i64) workgroup(%[[WG:.*]] : memref<1xf32, #gpu.address_space<workgroup>> {llvm.align = 16 : i32}) private(%[[PRIVATE:.*]] : memref<1xf32, #gpu.address_space<private>> {llvm.align = 16 : i32}) {
// CHECK: %{{.*}} = fir.emboxchar %[[CHAR]], %[[LENGTH]] : (!fir.ref<!fir.char<1,?>>, i64) -> !fir.boxchar<1>
// CHECK: memref.store %{{.*}}, %[[WG]][%{{.*}}] : memref<1xf32, #gpu.address_space<workgroup>>
// CHECK: memref.store %{{.*}}, %[[PRIVATE]][%{{.*}}] : memref<1xf32, #gpu.address_space<private>>
    
    gpu.func @test_with_char_proc(%arg0: f32, %arg1: tuple<() -> (), i64> {fir.char_proc}) workgroup(%arg2 : memref<1xf32, #gpu.address_space<workgroup>>) {
      %c0 = arith.constant 0 : index
      memref.store %arg0, %arg2[%c0] : memref<1xf32, #gpu.address_space<workgroup>>
      gpu.return
    }
// CHECK-LABEL: gpu.func @test_with_char_proc(
// CHECK-SAME: %{{.*}}: f32, %[[CHARPROC:.*]]: () -> () {fir.char_proc}, %[[LENGTH:.*]]: i64) workgroup(%[[WG:.*]] : memref<1xf32, #gpu.address_space<workgroup>>) {
// CHECK: %{{.*}} = fir.undefined tuple<() -> (), i64>
// CHECK: %{{.*}} = fir.insert_value %{{.*}}, %[[CHARPROC]], [0 : index] : (tuple<() -> (), i64>, () -> ()) -> tuple<() -> (), i64>
// CHECK: %{{.*}} = fir.insert_value %{{.*}}, %[[LENGTH]], [1 : index] : (tuple<() -> (), i64>, i64) -> tuple<() -> (), i64>
// CHECK: memref.store %{{.*}}, %[[WG]][%{{.*}}] : memref<1xf32, #gpu.address_space<workgroup>>
  }
}

// -----

module attributes {gpu.container_module, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  gpu.module @testmod {
    gpu.func @_QPtest(%arg0: complex<f32>) -> () kernel {
      gpu.return
    }
  }
  func.func @main(%arg0: complex<f32>) {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = fir.alloca i64
    %3 = cuf.stream_cast %2 : !fir.ref<i64>
    %4 = gpu.launch_func async [%3] @testmod::@_QPtest blocks in (%0, %0, %0) threads in (%0, %0, %0) : i64 dynamic_shared_memory_size %1 args(%arg0 : complex<f32>) {cuf.proc_attr = #cuf.cuda_proc<global>}
    return
  }
}

// CHECK-LABEL: func.func @main
// CHECK: %{{.*}} = gpu.launch_func async [%{{.*}}] @testmod::@_QPtest blocks in (%{{.*}}, %{{.*}}, %{{.*}}) threads in (%{{.*}}, %{{.*}}, %{{.*}}) : i64 dynamic_shared_memory_size %{{.*}} args(%{{.*}} : !fir.vector<2:f32>) {cuf.proc_attr = #cuf.cuda_proc<global>}

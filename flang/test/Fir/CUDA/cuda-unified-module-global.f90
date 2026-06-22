// End-to-end check that under -gpu=mem:unified, a plain host module-scope
// variable referenced from a global kernel:
//   1. is mirrored into the GPU module by CUFDeviceGlobal as a no-body
//      external declaration (so PTX gets `.extern .global ...`); and
//   2. is registered with the CUDA driver via
//      _FortranACUFRegisterExternalVariable (= __cudaRegisterHostVar) from
//      __cudaFortranConstructor, so the device-side symbol is mapped to
//      the host-resident storage at module-load time and HMM/ATS handles
//      migration.
//
// Pipeline: cuf-device-global with cuda-unified=true (clones the host
// global into the GPU module as an external declaration), then
// cuf-add-constructor with cuda-unified=true (emits the registration call
// for the cloned global).

// RUN: fir-opt --cuf-device-global="cuda-unified=true" --cuf-add-constructor="cuda-unified=true" %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i1 = dense<8> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, f80 = dense<128> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, i64 = dense<64> : vector<2xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, f128 = dense<128> : vector<2xi64>, !llvm.ptr<270> = dense<32> : vector<4xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, "dlti.stack_alignment" = 128 : i64, "dlti.endianness" = "little">, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", gpu.container_module, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  fir.global @_QMmtestsEm(dense<[1, 2, 3, 4, 5]> : tensor<5xi32>) : !fir.array<5xi32>

  func.func @_QMmtestsPg1() attributes {cuf.proc_attr = #cuf.cuda_proc<global>} {
    %0 = fir.address_of(@_QMmtestsEm) : !fir.ref<!fir.array<5xi32>>
    return
  }

  gpu.module @cuda_device_mod {
    gpu.func @_QMmtestsPg1() kernel {
      gpu.return
    }
  }
}

// Host-side definition is preserved.
// CHECK: fir.global @_QMmtestsEm(dense<[1, 2, 3, 4, 5]> : tensor<5xi32>) : !fir.array<5xi32>

// GPU module gets an external declaration (no body, no init). PTX lowers
// it as `.extern .global ...`; nvlink permits the extern because acclnk
// is invoked with -unifiedmem -init=unified -cudalink. The constructor
// below registers the host pointer via the CUDA driver.
// CHECK: gpu.module @cuda_device_mod
// CHECK: fir.global @_QMmtestsEm : !fir.array<5xi32>
// CHECK-NOT: fir.global @_QMmtestsEm{{.*}}dense

// Constructor registers the host pointer.
// CHECK: llvm.func internal @__cudaFortranConstructor()
// CHECK: cuf.register_module @cuda_device_mod -> !llvm.ptr
// CHECK: fir.address_of(@_QMmtestsEm) : !fir.ref<!fir.array<5xi32>>
// CHECK: fir.call @_FortranACUFRegisterExternalVariable
// CHECK-NOT: fir.call @_FortranACUFInitModule

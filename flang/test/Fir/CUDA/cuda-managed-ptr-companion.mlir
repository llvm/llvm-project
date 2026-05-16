// RUN: fir-opt --split-input-file --cuf-add-constructor %s | FileCheck %s --check-prefix=CONSTRUCTOR
// RUN: fir-opt --split-input-file --compiler-generated-names %s | FileCheck %s --check-prefix=NAMES

// Test 1 (CONSTRUCTOR): CUFAddConstructor attaches the cuf.managed_ptr unit
// attribute to the companion pointer global it creates for non-allocatable
// managed variables. This attribute is required so that downstream passes
// (CompilerGeneratedNamesConversionPass) can identify and exempt the global.

// Test 2 (NAMES): CompilerGeneratedNamesConversionPass does NOT rename a
// global that carries the cuf.managed_ptr attribute, leaving the dotted
// companion name intact so that CUFOpConversionLate can look it up.

// -----

// CONSTRUCTOR-LABEL: fir.global internal @_QMtestEmanx.managed.ptr
// CONSTRUCTOR-SAME:    cuf.managed_ptr
// CONSTRUCTOR-SAME:    section = "__nv_managed_data__"

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f32, dense<32> : vector<2xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", gpu.container_module, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"} {

  fir.global @_QMtestEmanx {data_attr = #cuf.cuda<managed>} : !fir.array<100xi32> {
    %0 = fir.zero_bits !fir.array<100xi32>
    fir.has_value %0 : !fir.array<100xi32>
  }

  gpu.module @cuda_device_mod {
    gpu.func @_QMtestPkernel() kernel {
      gpu.return
    }
    fir.global @_QMtestEmanx {data_attr = #cuf.cuda<managed>} : !fir.array<100xi32> {
      %0 = fir.zero_bits !fir.array<100xi32>
      fir.has_value %0 : !fir.array<100xi32>
    }
  }
}

// -----

// A companion pointer global bearing cuf.managed_ptr must not have its name
// mangled by CompilerGeneratedNamesConversionPass. Without the exemption the
// dots would be replaced with 'X', breaking the lookup in CUFOpConversionLate.

// NAMES:     fir.global internal @_QMtestEmanx.managed.ptr
// NAMES-NOT: @_QMtestEmanxXmanagedXptr

module {
  fir.global internal @_QMtestEmanx.managed.ptr {cuf.managed_ptr, section = "__nv_managed_data__"} : !fir.llvm_ptr<i8> {
    %0 = fir.zero_bits !fir.llvm_ptr<i8>
    fir.has_value %0 : !fir.llvm_ptr<i8>
  }
}

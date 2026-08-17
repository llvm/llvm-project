// RUN: fir-opt --cuf-add-constructor %s | FileCheck %s

// CUFAddConstructor must size a managed global of a derived type with a
// descriptor (allocatable/pointer) component instead of aborting.

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", gpu.container_module, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  fir.global @_QMmEobj {data_attr = #cuf.cuda<managed>} : !fir.array<10x!fir.type<_QMmTt{nodl:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>

  gpu.module @cuda_device_mod {
    fir.global @_QMmEobj {data_attr = #cuf.cuda<managed>} : !fir.array<10x!fir.type<_QMmTt{nodl:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>
  }
}

// CHECK-LABEL: llvm.func internal @__cudaFortranConstructor()
// CHECK: %[[SIZE:.*]] = arith.constant 480 : index
// CHECK: %[[SIZE_I64:.*]] = fir.convert %[[SIZE]] : (index) -> i64
// CHECK: fir.call @_FortranACUFRegisterManagedVariable(%{{.*}}, %{{.*}}, %{{.*}}, %[[SIZE_I64]])

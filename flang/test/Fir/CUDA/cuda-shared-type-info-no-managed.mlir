// A unit defining shared type information but no managed variable or device
// code still registers the pages of the section. The registration runs in the
// initialization constructor, which is terminated and listed in the global
// constructors.

// The global ending the type information is aligned to the largest page size
// of the target.

// RUN: fir-opt --cuf-shared-type-info --cuf-add-constructor="cuda-managed-type-info=true" %s | FileCheck %s --check-prefixes=CHECK,X86
// RUN: sed 's/x86_64-unknown-linux-gnu/aarch64-unknown-linux-gnu/' %s | fir-opt --cuf-shared-type-info | FileCheck %s --check-prefix=LARGEPAGE
// RUN: sed 's/x86_64-unknown-linux-gnu/powerpc64le-unknown-linux-gnu/' %s | fir-opt --cuf-shared-type-info | FileCheck %s --check-prefix=LARGEPAGE

module attributes {dlti.dl_spec = #dlti.dl_spec<i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i1 = dense<8> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, f80 = dense<128> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, i64 = dense<64> : vector<2xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, f128 = dense<128> : vector<2xi64>, !llvm.ptr<270> = dense<32> : vector<4xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, "dlti.stack_alignment" = 128 : i64, "dlti.endianness" = "little">, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  fir.global linkonce_odr @_QMmE.n.t constant : !fir.char<1> {
    %0 = fir.string_lit "t"(1) : !fir.char<1>
    fir.has_value %0 : !fir.char<1>
  }
}

// X86: llvm.mlir.global internal @__nv_type_info_end_{{[0-9a-f]+}}(0 : i8) {{.*}}alignment = 4096
// LARGEPAGE: llvm.mlir.global internal @__nv_type_info_end_{{[0-9a-f]+}}(0 : i8) {{.*}}alignment = 65536

// CHECK-LABEL: llvm.func internal @__cudaFortranConstructor()
// CHECK-NOT: CUFRegisterHostMemoryRange
// CHECK-NOT: CUFInitModule
// CHECK: llvm.return

// CHECK-LABEL: llvm.func internal @__cudaFortranInitConstructor()
// CHECK-NOT: CUFInitModule
// CHECK: %[[START:.*]] = llvm.mlir.addressof @__start___nv_type_info : !llvm.ptr
// CHECK: %[[STOP:.*]] = llvm.mlir.addressof @__stop___nv_type_info : !llvm.ptr
// CHECK: llvm.call @_FortranACUFRegisterHostMemoryRange(%[[START]], %[[STOP]]) : (!llvm.ptr, !llvm.ptr) -> ()
// CHECK-NEXT: llvm.return

// CHECK: llvm.mlir.global_ctors ctors = [@__cudaFortranConstructor, @__cudaFortranInitConstructor], priorities = [0 : i32, 1 : i32]

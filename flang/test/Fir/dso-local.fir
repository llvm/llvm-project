// RUN: fir-opt --split-input-file --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" %s | FileCheck %s

// Static relocation model, ELF, external definition: dso_local
module attributes {fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "",
    fir.relocation_model = 0 : i32,
    llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  fir.global @static_def : i32 {
    %0 = fir.zero_bits i32
    fir.has_value %0 : i32
  }
}
// CHECK: llvm.mlir.global external @static_def()
// CHECK-SAME: dso_local

// -----

// PIC + PIE, ELF, external definition: dso_local
module attributes {fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "",
    fir.relocation_model = 1 : i32, fir.is_pie,
    llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  fir.global @pie_def : i32 {
    %0 = fir.zero_bits i32
    fir.has_value %0 : i32
  }
}
// CHECK: llvm.mlir.global external @pie_def()
// CHECK-SAME: dso_local

// -----

// PIC without PIE, ELF, external definition: no dso_local
module attributes {fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "",
    fir.relocation_model = 1 : i32,
    llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  fir.global @pic_def : i32 {
    %0 = fir.zero_bits i32
    fir.has_value %0 : i32
  }
}
// CHECK: llvm.mlir.global external @pic_def()
// CHECK-NOT: dso_local
// CHECK-SAME: : i32

// -----

// Static relocation model, ELF, declaration only: no dso_local
module attributes {fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "",
    fir.relocation_model = 0 : i32,
    llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  fir.global @decl_only : i32
}
// CHECK: llvm.mlir.global external @decl_only()
// CHECK-NOT: dso_local
// CHECK-SAME: : i32

// -----

// Static relocation model, ELF, internal linkage: no dso_local from this path
module attributes {fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "",
    fir.relocation_model = 0 : i32,
    llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  fir.global internal @internal_var : i32 {
    %0 = fir.zero_bits i32
    fir.has_value %0 : i32
  }
}
// CHECK: llvm.mlir.global internal @internal_var()
// CHECK-NOT: dso_local
// CHECK-SAME: : i32

// -----

// Static relocation model, ELF, linkonce: no dso_local from this path
module attributes {fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "",
    fir.relocation_model = 0 : i32,
    llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  fir.global linkonce @linkonce_str constant : !fir.char<1,5> {
    %0 = fir.string_lit "hello"(5) : !fir.char<1,5>
    fir.has_value %0 : !fir.char<1,5>
  }
}
// CHECK: llvm.mlir.global linkonce constant @linkonce_str()
// CHECK-NOT: dso_local
// CHECK-SAME: : !llvm.array<5 x i8>

// -----

// Static relocation model, ELF, CUDA constant variable: no dso_local
// CUDA device/constant/managed variables need GOT indirection because the
// CUDA runtime interposes on these symbols via cudaRegisterVar.
module attributes {fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "",
    fir.relocation_model = 0 : i32,
    llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  fir.global @cuda_constant {data_attr = #cuf.cuda<constant>} : i32 {
    %0 = arith.constant 4 : i32
    fir.has_value %0 : i32
  }
}
// CHECK: llvm.mlir.global external @cuda_constant()
// CHECK-NOT: dso_local
// CHECK-SAME: : i32

// -----

// Static relocation model, ELF, CUDA device variable: no dso_local
module attributes {fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "",
    fir.relocation_model = 0 : i32,
    llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  fir.global @cuda_device {data_attr = #cuf.cuda<device>} : i32 {
    %0 = fir.zero_bits i32
    fir.has_value %0 : i32
  }
}
// CHECK: llvm.mlir.global external @cuda_device()
// CHECK-NOT: dso_local
// CHECK-SAME: : i32

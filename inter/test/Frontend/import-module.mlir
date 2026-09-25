// RUN: inter-opt %s --inter-import-llvm | FileCheck %s --check-prefix=IMPORT
// RUN: inter-opt %s --inter-import-llvm --inter-convert-llvm-to-xw | FileCheck %s --check-prefix=CLOSED

module attributes {
  dlti.dl_spec = #dlti.dl_spec<
    !llvm.ptr = dense<64> : vector<4xi64>,
    "dlti.endianness" = "little"
  >,
  llvm.keep = "target-independent",
  llvm.module_asm = [],
  llvm.target_triple = "spir64-unknown-unknown"
} {
}

// IMPORT: dlti.dl_spec
// IMPORT-SAME: llvm.keep = "target-independent"
// IMPORT-SAME: llvm.module_asm = []
// IMPORT-SAME: llvm.target_triple = "spir64-unknown-unknown"

// CLOSED-NOT: dlti.dl_spec
// CLOSED: llvm.keep = "target-independent"
// CLOSED-NOT: llvm.module_asm
// CLOSED-NOT: llvm.target_triple

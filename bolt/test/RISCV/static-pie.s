/// Check BOLT's handling of a static PIE binary (ET_DYN with PT_DYNAMIC
/// but no PT_INTERP) on RISC-V. On RV32 BOLT currently only supports
/// statically linked, non-PIE binaries, so the input must be rejected
/// with a clear error. On RV64 the same input is supported and BOLT
/// must not emit the RV32 error.

/// RV32 build and check.
// RUN: llvm-mc -triple riscv32 -filetype=obj -o %t.rv32.o %s
// RUN: ld.lld -pie -q -o %t.rv32 %t.rv32.o
// RUN: not llvm-bolt -o %t.rv32.bolted %t.rv32 2>&1 \
// RUN:     | FileCheck --check-prefix=RV32 %s

/// RV64 build and check.
// RUN: llvm-mc -triple riscv64 -filetype=obj -o %t.rv64.o %s
// RUN: ld.lld -pie -q -o %t.rv64 %t.rv64.o
// RUN: llvm-bolt -o %t.rv64.bolted %t.rv64 2>&1 \
// RUN:     | FileCheck --check-prefix=RV64 %s

// RV32: BOLT-ERROR: RV32 support is currently limited to statically linked, non-PIE binaries

// RV64: BOLT-INFO: Target architecture: riscv64
// RV64-NOT: BOLT-ERROR: RV32 support

  .text
  .globl _start
  .p2align 1
_start:
  ret
  .size _start, .-_start

/// Check BOLT's handling of a non-PIE, dynamically linked binary on
/// RISC-V. On RV32 BOLT currently only supports statically linked,
/// non-PIE binaries, so the input must be rejected with a clear error.
/// On RV64 the same input is supported and BOLT must not emit the RV32
/// error.

// RUN: split-file %s %t.dir

/// RV32 build and check.
// RUN: llvm-mc -triple riscv32 -filetype=obj -o %t.dir/main32.o %t.dir/main.s
// RUN: llvm-mc -triple riscv32 -filetype=obj -o %t.dir/lib32.o %t.dir/lib.s
// RUN: ld.lld -shared -soname libfoo.so -o %t.dir/libfoo32.so %t.dir/lib32.o
// RUN: ld.lld --no-pie -dynamic-linker /lib/ld.so.1 \
// RUN:     -L%t.dir %t.dir/libfoo32.so -o %t.rv32 %t.dir/main32.o
// RUN: not llvm-bolt -o %t.rv32.bolted %t.rv32 2>&1 \
// RUN:     | FileCheck --check-prefix=RV32 %s

/// RV64 build and check.
// RUN: llvm-mc -triple riscv64 -filetype=obj -o %t.dir/main64.o %t.dir/main.s
// RUN: llvm-mc -triple riscv64 -filetype=obj -o %t.dir/lib64.o %t.dir/lib.s
// RUN: ld.lld -shared -soname libfoo.so -o %t.dir/libfoo64.so %t.dir/lib64.o
// RUN: ld.lld --no-pie -dynamic-linker /lib/ld.so.1 \
// RUN:     -L%t.dir %t.dir/libfoo64.so -o %t.rv64 %t.dir/main64.o
// RUN: llvm-bolt -o %t.rv64.bolted %t.rv64 2>&1 \
// RUN:     | FileCheck --check-prefix=RV64 %s

// RV32: BOLT-ERROR: RV32 support is currently limited to statically linked, non-PIE binaries

// RV64: BOLT-INFO: Target architecture: riscv64
// RV64-NOT: BOLT-ERROR: RV32 support

//--- main.s
  .text
  .globl _start
  .p2align 1
_start:
  call foo
  ret
  .size _start, .-_start

//--- lib.s
  .text
  .globl foo
  .p2align 1
foo:
  ret
  .size foo, .-foo

// RUN: llvm-mc -triple riscv64 -filetype=obj -o %t.o %s
// RUN: ld.lld --emit-relocs -static -o %t.exe %t.o
// RUN: llvm-bolt --print-cfg --print-only=_start -o %t.bolt %t.exe \
// RUN:   | FileCheck %s

/// GNU ld can relax a tail call to an undefined weak symbol to
/// `jalr zero, zero, 0`, while preserving the R_RISCV_CALL_PLT relocation on
/// the preceding AUIPC. The JALR target is evaluatable as zero, but the
/// instruction is still an indirect branch and must not be rewritten as a
/// direct branch.

  .text
  .option norvc
  .weak weak_func

// CHECK-LABEL: Binary Function "_start
// CHECK:      auipc t1,
// CHECK-NEXT: jr zero # TAILCALL

  .globl _start
  .type _start, @function
_start:
  li a0, 0
  beqz a0, .Lreturn
.Lcall:
  auipc t1, 0
  .reloc .Lcall, R_RISCV_CALL_PLT, weak_func
  jalr zero, zero, 0
.Lreturn:
  ret

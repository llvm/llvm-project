## Check that BOLT recognizes a non-preemptible IFUNC IPLT entry and tracks an
## otherwise unnamed resolver after the linker canonicalizes the exported IFUNC
## symbol to the IPLT entry. Function sizes model normal compiler output, while
## discarding local symbols removes the resolver's remaining name. Moving the
## synthesized resolver also verifies that the IRELATIVE addend is updated.

# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+relax -o %t.64.o %s
# RUN: ld.lld -q -o %t.64 %t.64.o
# RUN: llvm-objcopy --discard-all %t.64
# RUN: llvm-bolt %t.64 -o %t.64.bolt --use-old-text=0 --lite=0 \
# RUN:   --print-disasm --print-only=_start 2>&1 | FileCheck %s \
# RUN:   --check-prefix=BOLT \
# RUN:   --implicit-check-not="Expected BF to be presented as IFUNC resolver"
# RUN: llvm-readelf -Wr -Ws %t.64.bolt > %t.64.dump
# RUN: llvm-objdump -d --no-show-raw-insn %t.64.bolt >> %t.64.dump
# RUN: FileCheck %s --input-file=%t.64.dump \
# RUN:   --check-prefixes=ELF,IPLT,RESOLVER

# BOLT: Binary Function "_start
# BOLT: auipc a0, %pcrel_hi(__BOLT_IFUNC_RESOLVERat{{[0-9a-f]+}}@PLT)

# ELF: R_RISCV_IRELATIVE {{ *}}[[#%x,RESOLVER:]]
# ELF: {{[0-9a-f]+}} 4 FUNC {{.*}} func

# IPLT: Disassembly of section .iplt:
# IPLT: <ifunc0>:
# IPLT-NEXT: {{.*}} auipc t3,
# IPLT-NEXT: {{.*}} ld t3,

# RESOLVER: {{^ *}}[[#%x,RESOLVER]]:{{ *}}ret

  .text
  .globl _start
  .type _start, @function
_start:
1:
  auipc a0, %pcrel_hi(ifunc0)
  addi a0, a0, %pcrel_lo(1b)
  .size _start, .-_start

  .globl func
  .type func, @function
func:
  ret
  .size func, .-func

  .globl ifunc0
  .type ifunc0, @gnu_indirect_function
ifunc0:
  ret
  .size ifunc0, .-ifunc0

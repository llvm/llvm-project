# REQUIRES: x86
# RUN: llvm-mc %s -o %t.o -filetype=obj -triple=x86_64-pc-linux

## GCC -ffunction-sections -fexceptions gives each function a
## .gcc_except_table.<func> that is not in a group and does not have
## SHF_LINK_ORDER. On RISC-V -mrelax the table contains R_RISCV_SET_ULEB128 /
## R_RISCV_SUB_ULEB128 relocations back to the function. The .quad below is
## that edge on x86_64: the table points at the function, and the function
## calls an undefined symbol.
##
## --gc-sections drops both sections when the function is unreferenced, so the
## undefined symbol does not fail the link. Referencing the function keeps the
## table and still reports the undefined symbol.

# RUN: ld.lld --gc-sections --print-gc-sections -u live %t.o -o /dev/null | \
# RUN:   FileCheck %s --check-prefix=GC
# RUN: not ld.lld --gc-sections -u dead %t.o -o /dev/null 2>&1 | \
# RUN:   FileCheck %s --check-prefix=UNDEF

# GC-NOT: .text.live
# GC:     removing unused section {{.*}}.o:(.text.dead)
# GC:     removing unused section {{.*}}.o:(.gcc_except_table.dead)
# GC-NOT: .text.live

# UNDEF:      error: undefined symbol: missing
# UNDEF-NEXT: >>> referenced by {{.*}}:(.text.dead+{{.*}})

.globl live, dead
.section .text.live,"ax",@progbits
live:
  ret

.section .text.dead,"ax",@progbits
dead:
  .cfi_startproc
  call missing
  ret
  .cfi_lsda 0x1b,.Llsda
  .cfi_endproc

.section .gcc_except_table.dead,"a",@progbits
.Llsda:
  .quad dead

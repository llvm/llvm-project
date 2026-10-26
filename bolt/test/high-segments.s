## Check that --custom-allocation-vma keeps new code near the original text
## despite a segment at 1 TiB. BOLT aligns 0x700000 up to 0x800000.

# REQUIRES: target-x86_64 || target-aarch64

# RUN: split-file %s %t
# RUN: %clang %cflags -c %t/main.s -o %t.o
# RUN: ld.lld -q -T %t/main.ls %t.o -o %t.exe
# RUN: llvm-bolt --custom-allocation-vma=0x700000 \
# RUN:   %t.exe -o %t.bolt 2>&1 | FileCheck %s

# CHECK: BOLT-INFO: creating new program header table at address 0x800000,
# CHECK: BOLT-INFO: enabling relocation mode

//--- main.s
  .section .my.reserved.section,"awx",@nobits
  .zero 0x80000000

  .text
  .globl _start
  .type _start,@function
  .reloc 0, BFD_RELOC_NONE      // AArch64 requires relocations
_start:
  ret
  .size _start, .-_start

//--- main.ls
SECTIONS
{
  .my.reserved.section 1<<40 : {
    *(.my.reserved.section);
  }
} INSERT BEFORE .comment;

# REQUIRES: x86
## Test that an explicit address expression on a .tbss section is respected,
## and that TLS offsets and PT_TLS are generated correctly when one is specified.

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 a.s -o a.o

# RUN: ld.lld -T explicit.t a.o -o explicit
# RUN: llvm-readelf -Sl explicit | FileCheck %s --check-prefixes=EXPLICIT,PHDR
# RUN: llvm-objdump -d --no-show-raw-insn explicit | FileCheck %s --check-prefix=DIS

## An explicit address on a .tbss output section is honored.
# EXPLICIT:      .tdata PROGBITS 0000000000100000 {{[0-9a-f]+}} 000004
# EXPLICIT:      .tbss  NOBITS   0000000000200000 {{[0-9a-f]+}} 000004

# PHDR: TLS 0x002000 0x0000000000100000 0x0000000000100000 0x000004 0x100004 R   0x1

# DIS:      movl	%fs:-0x100004, %eax
# DIS-NEXT: movl	%fs:-0x4, %eax

#--- a.s
.globl _start
_start:
  movl %fs:x@tpoff, %eax
  movl %fs:y@tpoff, %eax

.section .tdata,"awT",@progbits
.globl x
x:
  .long 1

.section .tbss,"awT",@nobits
.globl y
y:
  .long 0

#--- explicit.t
MEMORY {
  text_mem (rx) : ORIGIN = 0xFFFFFFFFFFF00000, LENGTH = 0x10000
  tdata_mem (rw) : ORIGIN = 0x100000, LENGTH = 0x1000
  tbss_mem (rw) : ORIGIN = 0x200000, LENGTH = 0x1000
}

SECTIONS {
  .text : { *(.text) } >text_mem AT>text_mem
  .tdata 0x100000 : { *(.tdata) } >tdata_mem AT>tdata_mem
  .tbss 0x200000 (NOLOAD) : { *(.tbss) } >tbss_mem AT>tbss_mem
}

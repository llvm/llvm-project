# REQUIRES: x86
## A branch from small code into a SHF_X86_64_LARGE section keeps an indirection,
## because the callee may be more than 2GB away.

# RUN: split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 a.s -o a.o

# RUN: ld.lld a.o -T lds -o a
# RUN: llvm-objdump -d --no-show-raw-insn a | FileCheck %s --check-prefix=DISASM
# RUN: llvm-readelf -x .got.plt a | FileCheck %s --check-prefix=GOTPLT
# RUN: llvm-readelf -r a | FileCheck %s --check-prefix=NOREL

# RUN: ld.lld -pie a.o -T lds -o a.pie
# RUN: llvm-readelf -rW a.pie | FileCheck %s --check-prefix=PIE

## The call to large targets an IPLT entry rather than large itself, while the
## call to small is relaxed to a direct branch as usual.
# DISASM:      <_start>:
# DISASM-NEXT:   callq 0x210000{{$}}
# DISASM-NEXT:   callq 0x200000 <small>

## The slot holds large's address (0x300000).
# GOTPLT:      Hex dump of section '.got.plt':
# GOTPLT-NEXT: 0x00220000 00003000 00000000

## Nothing is left for a dynamic linker to resolve.
# NOREL: There are no relocations in this file.

## Under PIC the slot only needs the load base added.
# PIE: 0000000000220000 0000000000000008 R_X86_64_RELATIVE 300000

#--- a.s
.section .ltext,"axl",@progbits
.globl large
.type large, @function
large:
  retq

.text
.globl small
.type small, @function
small:
  retq

.globl _start
.type _start, @function
_start:
  call large
  call small
  retq

#--- lds
SECTIONS {
  .text     0x200000 : { *(.text) }
  .iplt     0x210000 : { *(.iplt) }
  .got.plt  0x220000 : { *(.got.plt) }
  .ltext    0x300000 : { *(.ltext) }
}

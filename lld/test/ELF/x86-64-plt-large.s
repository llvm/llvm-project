# REQUIRES: x86
## A branch from small code into a SHF_X86_64_LARGE section keeps its PLT entry,
## because the callee may be more than 2GB away.

# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/a.s -o %t/a.o

# RUN: ld.lld %t/a.o -T %t/lds -o %t/a
# RUN: llvm-objdump -d --no-show-raw-insn %t/a | FileCheck %s --check-prefix=DISASM
# RUN: llvm-readelf -x .got.plt %t/a | FileCheck %s --check-prefix=GOTPLT
# RUN: llvm-readelf -r %t/a | FileCheck %s --check-prefix=NOREL

# RUN: ld.lld -pie %t/a.o -T %t/lds -o %t/a.pie
# RUN: llvm-readelf -r %t/a.pie | FileCheck %s --check-prefix=PIE

## The call to large targets a PLT entry rather than large itself, while the
## call to small is relaxed to a direct branch as usual.
# DISASM:      <_start>:
# DISASM-NEXT:   callq 0x210010{{$}}
# DISASM-NEXT:   callq 0x200000 <small>

## The slot holds large (0x300000) rather than a lazy binding stub.
# GOTPLT:      Hex dump of section '.got.plt':
# GOTPLT-NEXT: 0x00220000 00000000 00000000 00000000 00000000
# GOTPLT-NEXT: 0x00220010 00000000 00000000 00003000 00000000

## Nothing is left for a dynamic linker to resolve.
# NOREL: There are no relocations in this file.

## Under PIC the slot only needs the load base added, so the same slot gets a
## relative relocation whose addend is large, rather than a symbol lookup.
# PIE-NOT: R_X86_64_JUMP_SLOT
# PIE:     0000000000220018 0000000000000008 R_X86_64_RELATIVE 300000
# PIE-NOT: R_X86_64_JUMP_SLOT

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
  .text    0x200000 : { *(.text) }
  .plt     0x210000 : { *(.plt) }
  .got.plt 0x220000 : { *(.got.plt) }
  .ltext   0x300000 : { *(.ltext) }
}

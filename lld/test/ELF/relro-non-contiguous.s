# REQUIRES: x86
## A SECTIONS command may interleave relro and non-relro sections. Emit one
## PT_GNU_RELRO segment for each contiguous run of relro sections. Each
## relro->non-relro boundary also starts a fresh PT_LOAD.

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 a.s -o a.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 b.s -o b.o
# RUN: ld.lld -shared -soname=b b.o -o b.so

## .dynamic and the copy relocation in .bss.rel.ro are relro, while the
## intervening .got.plt is not (no -z now).
# RUN: ld.lld a.o b.so -z relro -T a.lds -o out
# RUN: llvm-readelf -l out | FileCheck %s

## -z norelro => No PT_GNU_RELRO.
# RUN: ld.lld a.o b.so -z norelro -T a.lds -o out2
# RUN: llvm-readelf -l out2 | FileCheck %s --check-prefix=NORELRO

## Three relro runs separated by two non-relro sections produce three
## PT_GNU_RELRO and three writable PT_LOAD segments.
# RUN: llvm-mc -filetype=obj -triple=x86_64 c.s -o c.o
# RUN: ld.lld c.o -z relro -T c.lds -o out3
# RUN: llvm-readelf -l out3 | FileCheck %s --check-prefix=THREE

#             Type      Offset   VirtAddr           PhysAddr           FileSiz  MemSiz   Flg Align
# CHECK:      GNU_RELRO 0x001100 0x0000000000000100 0x0000000000000100 0x000100 0x000100 R   0x1
# CHECK-NEXT: GNU_RELRO 0x001220 0x0000000000001000 0x0000000000001000 0x000000 0x000004 R   0x1

# CHECK:      Section to Segment mapping:
# CHECK:      05     .dynamic {{$}}
# CHECK-NEXT: 06     .bss.rel.ro {{$}}

# NORELRO-NOT: GNU_RELRO

# THREE:      GNU_RELRO
# THREE-NEXT: GNU_RELRO
# THREE-NEXT: GNU_RELRO
# THREE-NOT:  GNU_RELRO

# THREE:      Section to Segment mapping:
# THREE:      .data.rel.ro {{$}}
# THREE-NEXT: .w1 .jcr {{$}}
# THREE-NEXT: .w2 .init_array {{$}}

#--- a.lds
SECTIONS {
  .dynamic : { *(.dynamic) }
  .got.plt : { *(.got.plt) }
  . = ALIGN(CONSTANT(MAXPAGESIZE));
  .bss.rel.ro : { *(.bss.rel.ro) }
}

#--- a.s
.global _start
_start:
  .quad bar
  .quad foo

#--- b.s
.global bar
.type bar, %function
bar:
  nop

## foo is read-only in the DSO so the copy relocation lands in .bss.rel.ro.
.p2align 2
.type foo, @object
.global foo
foo:
.size foo, 4
.space 4

#--- c.lds
SECTIONS {
  .text        : { *(.text) }
  .data.rel.ro : { *(.data.rel.ro) }   ## relro run 1
  .w1          : { *(.w1) }
  .jcr         : { *(.jcr) }           ## relro run 2
  .w2          : { *(.w2) }
  .init_array  : { *(.init_array) }    ## relro run 3
}

#--- c.s
.global _start
_start:
  ret

.section .data.rel.ro,"aw",@progbits
.quad 0
.section .w1,"aw",@progbits
.quad 0
.section .jcr,"aw",@progbits
.quad 0
.section .w2,"aw",@progbits
.quad 0
.section .init_array,"aw",@init_array
.quad 0

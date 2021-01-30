# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o

## Show that we emit .rela.bar, .rela.text and .rela.debug_info when GC is disabled.
# RUN: ld.lld --emit-relocs %t.o -o %t
# RUN: llvm-objdump %t --section-headers | FileCheck %s --check-prefix=NOGC
# NOGC: .rela.text
# NOGC: .rela.bar
# NOGC: .rela.debug_info

## GC collects .bar section and we exclude .rela.bar from output. We keep
## .rela.text because we keep .text. We keep .rela.debug_info because we keep
## non-SHF_ALLOC .debug_info.
# RUN: ld.lld --gc-sections --emit-relocs --print-gc-sections %t.o -o %t \
# RUN:   | FileCheck --check-prefix=MSG %s
# MSG: removing unused section {{.*}}.o:(.bar)
# MSG: removing unused section {{.*}}.o:(.rela.bar)
# RUN: llvm-objdump %t --section-headers | FileCheck %s --check-prefix=GC --implicit-check-not=.rela.
# GC:      .rela.text
# GC-NEXT: .debug_info
# GC-NEXT: .rela.debug_info

.section .bar,"a"
.quad .bar

.text
relocs:
.quad _start

.global _start
_start:
 nop

.section .debug_info,"",@progbits
.quad .text

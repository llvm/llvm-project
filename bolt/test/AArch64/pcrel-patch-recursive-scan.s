## Ignoring an already-disassembled target recursively scans its references.
## The outer scan must still symbolize subsequent ADR and literal LDR operands.
## Cover both assembler-resolved references and a relocation-backed first
## reference, which takes the older ADR handling path.

# REQUIRES: system-linux

# RUN: llvm-mc -triple=aarch64 -filetype=obj %s -o %t.adr.o
# RUN: llvm-mc -triple=aarch64 -filetype=obj --defsym=GLOBAL=1 %s -o %t.adr-rel.o
# RUN: llvm-mc -triple=aarch64 -filetype=obj --defsym=LITERAL=1 %s -o %t.ldr.o
# RUN: llvm-mc -triple=aarch64 -filetype=obj --defsym=LITERAL=1 --defsym=GLOBAL=1 %s -o %t.ldr-rel.o

# RUN: llvm-readelf -r %t.adr.o | FileCheck %s --check-prefix=LOCAL
# RUN: llvm-readelf -r %t.ldr.o | FileCheck %s --check-prefix=LOCAL
# RUN: llvm-readelf -r %t.adr-rel.o | FileCheck %s --check-prefix=ADR-REL
# RUN: llvm-readelf -r %t.ldr-rel.o | FileCheck %s --check-prefix=LDR-REL
# LOCAL-NOT: R_AARCH64_ADR_PREL_LO21
# LOCAL-NOT: R_AARCH64_LD_PREL_LO19
# LOCAL: R_AARCH64_ABS64
# ADR-REL: R_AARCH64_ADR_PREL_LO21 {{.*}} first
# LDR-REL: R_AARCH64_LD_PREL_LO19 {{.*}} first

# RUN: ld.lld --emit-relocs -e _start %t.adr.o -o %t.adr
# RUN: ld.lld --emit-relocs -e _start %t.adr-rel.o -o %t.adr-rel
# RUN: ld.lld --emit-relocs -e _start %t.ldr.o -o %t.ldr
# RUN: ld.lld --emit-relocs -e _start %t.ldr-rel.o -o %t.ldr-rel
# RUN: llvm-bolt %t.adr -o %t.adr.bolt --skip-funcs=_start --trap-old-code
# RUN: llvm-bolt %t.adr-rel -o %t.adr-rel.bolt --skip-funcs=_start --trap-old-code
# RUN: llvm-bolt %t.ldr -o %t.ldr.bolt --skip-funcs=_start --trap-old-code
# RUN: llvm-bolt %t.ldr-rel -o %t.ldr-rel.bolt --skip-funcs=_start --trap-old-code

## Both targets must remain at their original addresses. In particular, keeping
## only the first target would leave the next reference pointing at old code.
# RUN: llvm-nm %t.adr | grep -E ' [tT] (first|second)$' > %t.adr.sym
# RUN: llvm-nm %t.adr.bolt | grep -E ' [tT] (first|second)$' | diff %t.adr.sym -
# RUN: llvm-nm %t.adr-rel | grep -E ' [tT] (first|second)$' > %t.adr-rel.sym
# RUN: llvm-nm %t.adr-rel.bolt | grep -E ' [tT] (first|second)$' | diff %t.adr-rel.sym -
# RUN: llvm-nm %t.ldr | grep -E ' [tT] (first|second)$' > %t.ldr.sym
# RUN: llvm-nm %t.ldr.bolt | grep -E ' [tT] (first|second)$' | diff %t.ldr.sym -
# RUN: llvm-nm %t.ldr-rel | grep -E ' [tT] (first|second)$' > %t.ldr-rel.sym
# RUN: llvm-nm %t.ldr-rel.bolt | grep -E ' [tT] (first|second)$' | diff %t.ldr-rel.sym -

  .text
  .ifdef GLOBAL
  .globl first
  .endif
  .type first, %function
first:
  ret
  .size first, .-first

  .type second, %function
second:
  mov w0, #42
  mov w8, #93
  svc #0
  .size second, .-second

  .globl _start
  .type _start, %function
_start:
  .ifdef LITERAL
  ldr w1, first
  ldr w0, second
  ret
  .else
  adr x1, first
  adr x0, second
  br x0
  .endif
  .size _start, .-_start

## Keep a text relocation even when both references are assembler-resolved.
  .quad _start

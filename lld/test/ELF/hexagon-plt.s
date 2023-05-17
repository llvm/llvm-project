# REQUIRES: hexagon
# RUN: echo '.globl bar, weak; .type bar,@function; .type weak,@function; bar: weak:' > %t1.s

# RUN: llvm-mc -filetype=obj -triple=hexagon-unknown-elf %t1.s -o %t1.o
# RUN: ld.lld -shared %t1.o -soname=t1.so -o %t1.so
# RUN: llvm-mc -mno-fixup -filetype=obj -triple=hexagon-unknown-elf %s -o %t.o
# RUN: ld.lld %t.o %t1.so -z separate-code -o %t
# RUN: llvm-readelf -S -s %t | FileCheck --check-prefixes=SEC,NM %s
# RUN: llvm-readobj -r %t | FileCheck --check-prefix=RELOC %s
# RUN: llvm-readelf -x .got.plt %t | FileCheck --check-prefix=GOTPLT %s
# RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t | FileCheck --check-prefixes=DIS %s

# SEC: .plt PROGBITS {{0*}}00020040

## A canonical PLT has a non-zero st_value. bar and weak are called but their
## addresses are not taken, so a canonical PLT is not necessary.
# NM: {{0*}}00000000 0 FUNC GLOBAL DEFAULT UND bar
# NM: {{0*}}00000000 0 FUNC WEAK   DEFAULT UND weak

## The .got.plt slots relocated by .rela.plt point to .plt
## This is required by glibc.
# RELOC:      .rela.plt {
# RELOC-NEXT:    0x40078 R_HEX_JMP_SLOT bar 0x0
# RELOC-NEXT:    0x4007C R_HEX_JMP_SLOT weak 0x0
# RELOC-NEXT: }
# GOTPLT:      section '.got.plt'
# GOTPLT-NEXT: 0x00040068 00000000 00000000 00000000 00000000
# GOTPLT-NEXT: 0x00040078 00000000 00000000

# DIS:      <_start>:
## Direct call
## Call foo directly
# DIS-NEXT:   { call 0x2003c }
## Call bar via plt
# DIS-NEXT:   { call 0x20060 }
## Call weak via plt
# DIS-NEXT:   { call 0x20070 }
# DIS-NEXT: { 	immext(#0)

## Call foo directly
# DIS-NEXT: if (p0) jump:nt 0x2003c }
# DIS-NEXT: { 	immext(#64)
## Call bar via plt
# DIS-NEXT: if (p0) jump:nt 0x20060 }
# DIS-NEXT: { 	immext(#64)
## Call weak via plt
# DIS-NEXT: if (p0) jump:nt 0x20070 }
# DIS-NEXT: { 	immext(#0)

## Call foo directly
# DIS-NEXT: r0 = #0 ; jump 0x2003c }
# DIS-NEXT: { 	immext(#0)
## Call bar via plt
# DIS-NEXT: r0 = #0 ; jump 0x20060 }
# DIS-NEXT: { 	immext(#0)
## Call weak via plt
# DIS-NEXT: r0 = #0 ; jump 0x20070 }

# DIS:      <foo>:
# DIS-NEXT:   2003c:


# DIS: Disassembly of section .plt:

# DIS: 00020040 <.plt>:
# DIS-NEXT:   20040: { 	immext(#131072)
# DIS-NEXT:   20044:   	r28 = add(pc,##131112) }
# DIS-NEXT:   20048: { 	r14 -= add(r28,#16)
# DIS-NEXT:   2004c:   	r15 = memw(r28+#8)
# DIS-NEXT:   20050:   	r28 = memw(r28+#4) }
# DIS-NEXT:   20054: { 	r14 = asr(r14,#2)
# DIS-NEXT:   20058:   	jumpr r28 }
# DIS-NEXT:   2005c: { 	trap0(#219) }
## bar's plt slot
# DIS-NEXT:   20060: { 	immext(#131072)
# DIS-NEXT:   20064:   	r14 = add(pc,##131096) }
# DIS-NEXT:   20068: { 	r28 = memw(r14+#0) }
# DIS-NEXT:   2006c: { 	jumpr r28 }
## weak's plt slot
# DIS-NEXT:   20070: { 	immext(#131072)
# DIS-NEXT:   20074:   	r14 = add(pc,##131084) }
# DIS-NEXT:   20078: { 	r28 = memw(r14+#0) }
# DIS-NEXT:   2007c: { 	jumpr r28 }


.global _start, foo, bar
.weak weak

_start:
  call foo
  call bar
  call weak
  if (p0) jump foo
  if (p0) jump bar
  if (p0) jump weak
  { r0 = #0; jump foo }
  { r0 = #0; jump bar }
  { r0 = #0; jump weak }

## foo is local and non-preemptale, no PLT is generated.
foo:
  jumpr r31
